"""Segment + motion selection: load_motion(segments=...) and cfg.segments threading."""
import numpy as np
import pytest
from t1_wbc._assets import asset
from t1_wbc.reference import load_motion, ReferenceTrajectory
from t1_wbc.model import load_t1_model, build_index_maps
from t1_wbc.config import WBCConfig

H50 = asset("motion_plan_h50cm.pkl")


def test_load_motion_selects_segment_subset():
    full = load_motion(H50)
    sub = load_motion(H50, segments=(0, 1))
    assert len(full) == 8
    assert len(sub) == 2
    # the selected subset is exactly the first two of the full motion, in order
    assert sub[0].T == full[0].T and sub[1].T == full[1].T
    np.testing.assert_array_equal(sub[0].left_arm, full[0].left_arm)
    np.testing.assert_array_equal(sub[1].trunk_xyz, full[1].trunk_xyz)


def test_load_motion_preserves_order_and_can_reorder():
    full = load_motion(H50)
    sub = load_motion(H50, segments=(2, 0))
    assert len(sub) == 2
    np.testing.assert_array_equal(sub[0].trunk_xyz, full[2].trunk_xyz)
    np.testing.assert_array_equal(sub[1].trunk_xyz, full[0].trunk_xyz)


def test_load_motion_rejects_out_of_range_segment():
    with pytest.raises((IndexError, ValueError)):
        load_motion(H50, segments=(0, 99))


def test_load_motion_none_segments_loads_all():
    assert len(load_motion(H50, segments=None)) == len(load_motion(H50))


def test_reference_trajectory_uses_cfg_segments():
    model, _ = load_t1_model()
    maps = build_index_maps(model)
    qhome = np.zeros(model.nu)
    cfg_full = WBCConfig(motion=H50)
    cfg_sub = WBCConfig(motion=H50, segments=(0, 1))
    ref_full = ReferenceTrajectory(model, maps, qhome, cfg_full)
    ref_sub = ReferenceTrajectory(model, maps, qhome, cfg_sub)
    assert 0.0 < ref_sub.duration < ref_full.duration
    # seg0+seg1 ref-time (~2.094s) times the playback scale, within seam-dedup tolerance
    segs = load_motion(H50, segments=(0, 1))
    expected = (segs[0].T * segs[0].dt + segs[1].T * segs[1].dt) * cfg_sub.time_scale
    assert abs(ref_sub.duration - expected) < 0.1 * expected
