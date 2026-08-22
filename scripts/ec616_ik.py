"""EC616 inverse kinematics bridge for act_robot → embody_model_eval.

Uses the in-repo ``ec616_kin`` package (vendored EA66 FK/IK; no external
``demo_test`` dependency).

act_robot raw / infer ``cartesian_abs`` uses TCP pose in base frame:
  - translation: meters
  - orientation: intrinsic xyz Euler, radians

Default tool offset: flange → TCP translation +0.18 m along flange Z.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# act_robot data uses TCP; EA66 default tool length ≈ 180 mm (validated vs FK).
DEFAULT_TCP_TOOL_Z_M = 0.18
CARTESIAN_EULER_SEQ = 'xyz'


@lru_cache(maxsize=1)
def _flange_tool_matrix(tcp_tool_z_m: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = float(tcp_tool_z_m)
    return T


def cartesian6_to_flange_T(
    cartesian6: np.ndarray,
    *,
    tcp_tool_z_m: float = DEFAULT_TCP_TOOL_Z_M,
    euler_seq: str = CARTESIAN_EULER_SEQ,
) -> np.ndarray:
    """TCP pose [x,y,z,rx,ry,rz] → 4×4 flange pose (meters)."""
    from scipy.spatial.transform import Rotation

    c6 = np.asarray(cartesian6, dtype=np.float64).reshape(6)
    T_tcp = np.eye(4, dtype=np.float64)
    T_tcp[:3, :3] = Rotation.from_euler(euler_seq, c6[3:6], degrees=False).as_matrix()
    T_tcp[:3, 3] = c6[:3]
    T_tool = _flange_tool_matrix(tcp_tool_z_m)
    return T_tcp @ np.linalg.inv(T_tool)


def solve_ik_cartesian_to_joint(
    cartesian6: np.ndarray,
    seed_joint6_rad: np.ndarray,
    *,
    tcp_tool_z_m: float = DEFAULT_TCP_TOOL_Z_M,
    enforce_soft_limits: bool = False,
    fallback_to_seed: bool = True,
) -> tuple[np.ndarray, bool]:
    """Cartesian TCP target → 6 joint angles (rad).

    Returns ``(joints_rad, success)``. On failure, returns seed when
    ``fallback_to_seed`` is True, otherwise re-raises.
    """
    from ec616_kin import ik_flange

    seed_rad = np.asarray(seed_joint6_rad, dtype=np.float64).reshape(6)
    seed_deg = np.degrees(seed_rad)
    T_flange = cartesian6_to_flange_T(cartesian6, tcp_tool_z_m=tcp_tool_z_m)

    lm_method = 'trf' if enforce_soft_limits else 'lm'
    try:
        out = ik_flange(
            T_flange,
            seed_deg,
            return_details=True,
            enforce_teach_soft_limits=enforce_soft_limits,
            lm_method=lm_method,
            max_nfev=800 if enforce_soft_limits else 400,
        )
        if out.success:
            return np.deg2rad(np.asarray(out.joint_deg, dtype=np.float64)), True
        msg = f'IK did not converge: {out.message} (residual={out.residual_norm:.3g})'
        if fallback_to_seed:
            return seed_rad.copy(), False
        raise RuntimeError(msg)
    except RuntimeError:
        if fallback_to_seed:
            return seed_rad.copy(), False
        raise
