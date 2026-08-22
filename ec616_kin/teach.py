from __future__ import annotations

"""示教器语义相关工具函数。

本模块聚焦“示教器数据解释层”，不直接参与 DH 链式运算：
- 软限位处理；
- 示教位姿结构化导出；
- 现场安装角与位姿格式转换。
"""

from typing import List, Literal, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import (
    TEACH_INSTALL_ROT_XB_Q1_DEG,
    TEACH_INSTALL_ROT_Z1_DEG,
    TEACH_JOINT_SOFT_MAX_DEG,
    TEACH_JOINT_SOFT_MIN_DEG,
    TEACH_PENDANT_POSES,
)


def teach_joint_soft_limits_deg() -> Tuple[np.ndarray, np.ndarray]:
    """返回示教器软限位（单位 deg）。

    Returns
    -------
    lo, hi:
        两个形状为 (6,) 的 ndarray，对应各关节最小/最大角。
    """
    lo = np.array(TEACH_JOINT_SOFT_MIN_DEG, dtype=float)
    hi = np.array(TEACH_JOINT_SOFT_MAX_DEG, dtype=float)
    return lo, hi


def clip_joints_to_teach_soft_limits(joint_deg: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """将关节角裁剪到软限位范围（单位 deg）。"""
    lo, hi = teach_joint_soft_limits_deg()
    return np.clip(np.asarray(joint_deg, dtype=float).reshape(6), lo, hi)


def joints_within_teach_soft_limits(joint_deg: Union[np.ndarray, Sequence[float]]) -> bool:
    """判断关节角是否位于软限位内（带微小容差）。"""
    q = np.asarray(joint_deg, dtype=float).reshape(6)
    lo, hi = teach_joint_soft_limits_deg()
    # 1e-9 容差用于抵抗浮点误差，避免边界点误判。
    return bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))


def teach_poses_joint_xyz_mm() -> List[Tuple[np.ndarray, np.ndarray]]:
    """导出示教标定数据为 ndarray 形式，便于数值计算。"""
    return [
        (np.array(p.joint_deg, dtype=float), np.array(p.position_mm, dtype=float))
        for p in TEACH_PENDANT_POSES
    ]


def teach_flange_T_m_from_pendant(
    position_mm: Union[np.ndarray, Sequence[float]],
    rxyz_deg: Union[np.ndarray, Sequence[float]],
    *,
    euler_seq: Literal["XYZ", "ZYX", "xyz", "zyx"] = "XYZ",
    degrees: bool = True,
) -> np.ndarray:
    """
    示教器报告的法兰位姿 → 4×4 **米**（基座→法兰），平移与 ``fk_flange`` 一致。

    Notes
    -----
    `Rotation.from_euler` 的欧拉序采用调用者可配置策略（默认 "XYZ"），
    以适配不同控制器/离线软件对 RX/RY/RZ 的定义差异。
    """
    p = np.asarray(position_mm, dtype=float).reshape(3)
    r = np.asarray(rxyz_deg, dtype=float).reshape(3)
    R = Rotation.from_euler(euler_seq, r, degrees=degrees).as_matrix()
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p / 1000.0
    return T


def teach_base_rotation_from_install_deg() -> np.ndarray:
    """根据安装节角度构造基座旋转矩阵。

    组合规则采用 ``Rz(V) @ Rx(Q1)``。当两角接近 0 时直接返回单位阵，
    以减少不必要的数值扰动。
    """
    if abs(TEACH_INSTALL_ROT_Z1_DEG) < 1e-12 and abs(TEACH_INSTALL_ROT_XB_Q1_DEG) < 1e-12:
        return np.eye(3, dtype=float)
    rz = np.deg2rad(TEACH_INSTALL_ROT_Z1_DEG)
    rx = np.deg2rad(TEACH_INSTALL_ROT_XB_Q1_DEG)
    c1, s1 = np.cos(rz), np.sin(rz)
    R_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c2, s2 = np.cos(rx), np.sin(rx)
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]], dtype=float)
    return R_z @ R_x
