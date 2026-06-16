"""Booster T1 MuJoCo loader + index maps. Verified vs mujoco 3.6.0."""
import mujoco
import numpy as np

from ._assets import asset

# Self-contained: the MJCF ships inside the package (see config.py / _assets.py).
T1_XML = asset("robot", "t1.xml")


def load_t1_model(xml=T1_XML):
    """Load the Booster T1 MJCF. Returns (model, data)."""
    model = mujoco.MjModel.from_xml_path(xml)
    return model, mujoco.MjData(model)


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
