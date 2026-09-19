"""SO-101 forward/inverse kinematics from the URDF joint chain (base_link -> gripper_tcp).

Pure numpy, no servo driver, so it runs on the PC. Angles are URDF joint radians, i.e. what
robot_control_5dof.joint_tick_to_rad returns (ZERO_TICKS 1990/2048/2048/2048/2126, direction +1).

  python3 kinematics.py          # self-check: FK sanity, IK round trip, a table target

Table frame (mm, see TABLE): origin = rear-left table corner seen from behind the arm,
+x along the 850 mm edge to the right, +y into the table (arm forward), +z up, z=0 table top.
TCP frame: +z = approach (out of the fingertips), +x = finger opening axis."""
from __future__ import annotations

import math

import numpy as np

# (xyz, rpy) of each joint origin in its parent link, from so101_ver1_original.urdf. All axes are z.
JOINTS = (
    ((0.0388353, 0.0, 0.0624), (math.pi, 0.0, -math.pi)),                  # 1 shoulder_pan  (base_link)
    ((-0.0303992, -0.0182778, -0.0542), (-math.pi / 2, -math.pi / 2, 0.0)),  # 2 shoulder_lift (shoulder_link)
    ((-0.11257, -0.028, 0.0), (0.0, 0.0, math.pi / 2)),                      # 3 elbow_flex    (upper_arm_link)
    ((-0.1349, 0.0052, 0.0), (0.0, 0.0, -math.pi / 2)),                      # 4 wrist_flex    (lower_arm_link)
    ((0.0, -0.0611, 0.0181), (math.pi / 2, 0.0486795, math.pi)),             # 5 wrist_roll    (wrist_link)
)
FIXED_TO_TCP = (
    ((-0.0269679069519, -0.00131401753426, -0.0801000900269), (1.5707954168329252, 3.628942180264e-06, 1.6194722907041545)),
    ((0.0, -0.0780187530518, 0.0270000119209), (math.pi / 2, 0.0, 0.0)),
)
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
RAD_LIMITS = ((-1.91986, 1.91986), (-1.74533, 1.74533), (-1.57079632679, 1.69), (-1.65806, 1.65806), (-2.743847297, 2.841206309))

# Installation measured 2026-09-18: base plate flush with the rear 850 mm edge, J1 axis 410 mm from the
# left short edge. Base plate rear edge is 22.4 mm behind base_link origin, J1 axis 38.8 mm ahead of it
# (mesh bounding boxes), so the axis sits 61.2 mm into the table. base_link z=0 is 2.4 mm above the plate bottom.
# ponytail: one fixed table pose; make this a JSON config when the arm moves or a second arm appears.
TABLE = {"j1_axis_xy_mm": (410.0, 61.2), "base_z_mm": 2.4}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _tf(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rpy(*rpy)
    t[:3, 3] = xyz
    return t


def _rz(q: float) -> np.ndarray:
    t = np.eye(4)
    c, s = math.cos(q), math.sin(q)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


_JOINT_TFS = [_tf(*j) for j in JOINTS]
_TCP_TF = _tf(*FIXED_TO_TCP[0]) @ _tf(*FIXED_TO_TCP[1])


def fk(q) -> np.ndarray:
    """4x4 pose of gripper_tcp in base_link for the 5 arm joint angles (rad)."""
    t = np.eye(4)
    for tf, qi in zip(_JOINT_TFS, q):
        t = t @ tf @ _rz(qi)
    return t @ _TCP_TF


def ik(target_m, approach=None, finger=None, q0=None, iters=300, tol_mm=0.5):
    """Joint angles placing the TCP at target_m (base_link, metres) with TCP +z along `approach`.
    approach=None constrains position only and, seeded with q0, returns the nearby solution (used for lifts).
    `finger` optionally aligns the TCP +x (finger opening axis); with 5 DOF it is satisfied only as far
    as the position/approach constraints leave room. Returns (q, residual_mm, ok)."""
    target = np.asarray(target_m, float)
    approach = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
    finger = None if finger is None else np.asarray(finger, float) / np.linalg.norm(finger)
    lo, hi = np.array(RAD_LIMITS).T
    seeds = [np.asarray(q0, float)] if q0 is not None else []
    pan = math.atan2(target[1], target[0] - 0.0388)
    seeds += [np.array([pan, 0.6, 1.0, -0.4, 0.0]), np.array([pan, 0.0, 0.0, 0.0, 0.0])]
    rng = np.random.default_rng(1)  # deterministic restarts for the local minima the two guesses miss
    seeds += [np.concatenate(([pan], rng.uniform(lo[1:] * 0.8, hi[1:] * 0.8))) for _ in range(8)]

    def residual(q):
        t = fk(q)
        parts = [(target - t[:3, 3]) * 10.0]                    # 1 mm -> 0.01
        if approach is not None:
            parts.append(np.cross(t[:3, 2], approach) * 0.1)   # 1 rad -> 0.1 (~0.1 weight vs 10 mm)
        if finger is not None and approach is not None:
            f = finger - approach * (approach @ finger)         # only the part perpendicular to approach
            parts.append(np.cross(t[:3, 0], f) * 0.02)
        return np.concatenate(parts)

    best = None
    for q in seeds:
        q = np.clip(q, lo, hi)
        lam = 1e-3
        for _ in range(iters):
            r = residual(q)
            jac = np.empty((r.size, 5))
            for k in range(5):
                dq = np.zeros(5)
                dq[k] = 1e-6
                jac[:, k] = (residual(q + dq) - r) / 1e-6
            step = np.linalg.solve(jac.T @ jac + lam * np.eye(5), -jac.T @ r)
            step = np.clip(step, -0.3, 0.3)
            q_new = np.clip(q + step, lo, hi)
            if np.linalg.norm(residual(q_new)) < np.linalg.norm(r):
                q, lam = q_new, max(lam * 0.5, 1e-6)
            else:
                lam = min(lam * 4, 1e2)
            if np.linalg.norm(step) < 1e-9:
                break
        t = fk(q)
        pos_mm = np.linalg.norm(target - t[:3, 3]) * 1000
        app_deg = 0.0 if approach is None else math.degrees(math.acos(np.clip(t[:3, 2] @ approach, -1, 1)))
        if best is None or pos_mm < best[1]:
            best = (q, pos_mm, app_deg)
        if pos_mm < tol_mm and app_deg < 2.0:
            return q, pos_mm, True
    return best[0], best[1], False


def base_from_table(p_mm) -> np.ndarray:
    """Table-frame point (mm) -> base_link (m). Arm forward (+x base) is +y table; base +y is -x table."""
    jx, jy = TABLE["j1_axis_xy_mm"]
    x, y, z = p_mm
    return np.array([(y - jy) / 1000 + 0.0388353, -(x - jx) / 1000, (z - TABLE["base_z_mm"]) / 1000])


def table_from_base(p_m) -> np.ndarray:
    """base_link point (m) -> table frame (mm)."""
    jx, jy = TABLE["j1_axis_xy_mm"]
    bx, by, bz = p_m
    return np.array([jx - by * 1000, jy + (bx - 0.0388353) * 1000, bz * 1000 + TABLE["base_z_mm"]])


def dir_base_from_table(v) -> np.ndarray:
    """Table-frame direction -> base_link direction (same rotation, no offset)."""
    x, y, z = v
    return np.array([y, -x, z], float)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    lo, hi = np.array(RAD_LIMITS).T
    # round trip on random reachable poses
    worst = 0.0
    for _ in range(20):
        q_true = rng.uniform(lo * 0.7, hi * 0.7)
        t = fk(q_true)
        q, err, ok = ik(t[:3, 3], t[:3, 2], finger=t[:3, 0])
        worst = max(worst, err)
        assert ok, f"ik failed on reachable pose: {err:.2f} mm"
    print(f"round trip: 20 poses, worst {worst:.3f} mm")
    # frame conventions
    assert np.allclose(table_from_base(base_from_table((100.0, 200.0, 30.0))), (100.0, 200.0, 30.0))
    assert np.allclose(table_from_base((0.0388353, 0.0, -0.0024)), (410.0, 61.2, 0.0)), "J1 axis foot must be the table point"
    # the hand is 219 mm from the wrist_flex axis, so a vertical approach never fits; 30-60 deg tilt toward
    # the arm reaches y 250-450 mm at z 40-120 mm (reachability sweep 2026-09-18). Check one of each.
    tilt = math.radians(45)
    q, err, ok = ik(base_from_table((410.0, 350.0, 80.0)), dir_base_from_table((0, math.sin(tilt), -math.cos(tilt))), finger=dir_base_from_table((1, 0, 0)))
    print("tilt45 (410,350,80):", " ".join(f"{n}={math.degrees(v):7.1f}" for n, v in zip(JOINT_NAMES, q)), f"err={err:.2f}mm ok={ok}")
    assert ok
    q, err, ok = ik(base_from_table((410.0, 400.0, 80.0)), dir_base_from_table((0, 1, 0)), finger=dir_base_from_table((1, 0, 0)))
    print("side   (410,400,80):", " ".join(f"{n}={math.degrees(v):7.1f}" for n, v in zip(JOINT_NAMES, q)), f"err={err:.2f}mm ok={ok}")
    assert ok
    q, err, ok = ik(base_from_table((410.0, 250.0, 80.0)), dir_base_from_table((0, 0, -1)))
    assert not ok, "straight-down at 250 mm should be out of reach; if it is not, the chain changed"
    # rest pose from the 2026-09-17 reading: must be low and behind/near the base, not out on the table
    rest = np.radians([-0.9, -105.0, 93.2, -93.0, 0.4])
    print("rest pose TCP (table mm):", np.round(table_from_base(fk(rest)[:3, 3]), 1))
    print("KINEMATICS_SELF_CHECK_OK")


if __name__ == "__main__":
    _self_check()
