"""Booster T1 MuJoCo loader + index maps. Verified vs mujoco 3.6.0."""
import mujoco
import numpy as np

T1_XML = ("/home/yoonwoo/humanoid_mpc_ws/src/t1_controller/robot_models/"
          "booster_t1/t1_description/urdf/t1.xml")


def load_t1_model(xml=T1_XML):
    """Load the Booster T1 MJCF. Returns (model, data)."""
    model = mujoco.MjModel.from_xml_path(xml)
    return model, mujoco.MjData(model)


# mujoco_warp sizes the per-world contact/constraint buffers from a model-only
# heuristic (T1: nconmax=48, njmax=64). The T1 double stance is 8 box-corner
# contacts (condim 3 -> 24 contact rows) + 29 joint friction-loss rows, so per-world
# ``nefc`` sits right at the default njmax=64 and, under per-env perturbation /
# dynamic balance, transiently spikes to ~78. When ``nefc > njmax`` mujoco_warp
# prints ``nefc overflow - please increase njmax`` and silently drops the overflow
# constraints (and the underlying contacts), which corrupts large-B (>=256) runs.
# Size both buffers explicitly with headroom (next valid bucket above the spike).
T1_NCONMAX = 64        # contacts/world: covers the 8 double-stance corners + margin
T1_NJMAX = 128         # constraints/world: covers ~63 steady nefc + perturbation spikes


def load_t1_warp_model(num_envs, xml=T1_XML):
    """Load the Booster T1 as a batched mjwarp model/data (B=num_envs).

    Forces dense qM (``mjJAC_DENSE`` BEFORE ``put_model`` -> ``wm.is_sparse`` False),
    so the QP can slice the padded ``qM`` to ``nv`` and ``mjw.jac`` runs dense.
    Sizes the contact/constraint buffers (``nconmax``/``njmax``) above the T1's
    double-stance ``nefc`` so large-B perturbed balance runs do not overflow and drop
    contacts. Returns ``(m, wm, wd, handles)``; every ``wp.array``/``wp.from_torch``
    in the batched dynamics must route through ``handles["wp_device"]``.
    """
    import mujoco_warp as mjw

    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE        # BEFORE put_model
    wm = mjw.put_model(m)
    assert wm.is_sparse is False                           # is_sparse lives on wm
    wd = mjw.put_data(m, mujoco.MjData(m), nworld=num_envs,
                      nconmax=T1_NCONMAX, njmax=T1_NJMAX)
    handles = build_warp_handles(m, num_envs, str(wd.qpos.device))
    return m, wm, wd, handles


def build_warp_handles(m, num_envs, wp_device):
    """Model-derived dynamics handles for ``WarpDynamics`` (engine-agnostic).

    Everything here is a pure function of the ``MjModel`` plus the warp device and the
    batch size. Both the standalone-Warp path (``load_t1_warp_model``) and the mjlab
    path (``build_mjlab_sim`` in controller.py) share this so the dynamics extraction
    is identical regardless of who owns the sim/stepping.
    """
    maps = build_index_maps(m)
    feet = maps["feet"]
    bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    return dict(
        nv=m.nv, nbody=m.nbody, nworld=num_envs,
        wp_device=str(wp_device),
        base_body=maps["base_body_id"],
        foot_L=feet["left"]["body_id"], foot_R=feet["right"]["body_id"],
        hand_L=bid("left_hand_link"), hand_R=bid("right_hand_link"),
        sole_local=feet["left"]["sole_local"].copy(),
        x_half=feet["left"]["x_half"], y_half=feet["left"]["y_half"],
        body_mass=m.body_mass.copy(), total_mass=float(m.body_mass.sum()),
        actuated_dof=np.array([j["dofadr"] for j in maps["actuated_joints"]], dtype=int),
        ctrlrange=m.actuator_ctrlrange.copy(), dof_frictionloss=m.dof_frictionloss.copy(),
    )


def build_index_maps(model):
    """Resolve joint/actuator/foot/hand/base handles for the T1 model as a dict."""
    n = lambda t, i: (mujoco.mj_id2name(model, t, i) or f"<unnamed#{i}>")
    OBJJ, OBJA, OBJB = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR,
                        mujoco.mjtObj.mjOBJ_BODY)
    free_jid = next(j for j in range(model.njnt)
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    base_bid = int(model.jnt_bodyid[free_jid])

    joint_id_to_act_id = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id_to_act_id[int(model.actuator_trnid[a, 0])] = a

    actuated = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = joint_id_to_act_id.get(j)
        actuated.append(dict(
            name=n(OBJJ, j), jnt_id=j,
            qposadr=int(model.jnt_qposadr[j]), dofadr=int(model.jnt_dofadr[j]),
            act_id=(int(a) if a is not None else None),
            ctrlrange=(model.actuator_ctrlrange[a].copy() if a is not None else None),
        ))

    def foot_handle(body_name):
        bid = mujoco.mj_name2id(model, OBJB, body_name)
        g0, gn = model.body_geomadr[bid], model.body_geomnum[bid]
        cg = next(g for g in range(g0, g0 + gn)
                  if model.geom_contype[g] != 0
                  and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
        sz, ps = model.geom_size[cg].copy(), model.geom_pos[cg].copy()
        return dict(body_id=bid, sole_local=ps.copy(),
                    x_half=float(sz[0]), y_half=float(sz[1]))

    return dict(
        free_joint_id=free_jid, base_body_id=base_bid, base_body_name=n(OBJB, base_bid),
        actuated_joints=actuated,
        joint_name_to_qposadr={d["name"]: d["qposadr"] for d in actuated},
        joint_name_to_dofadr={d["name"]: d["dofadr"] for d in actuated},
        name_to_act_index={d["name"]: i for i, d in enumerate(actuated)},
        feet={"left": foot_handle("left_foot_link"),
              "right": foot_handle("right_foot_link")},
        hands={"left": dict(body_id=mujoco.mj_name2id(model, OBJB, "left_hand_link")),
               "right": dict(body_id=mujoco.mj_name2id(model, OBJB, "right_hand_link"))},
    )
