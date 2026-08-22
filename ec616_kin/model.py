from __future__ import annotations

"""核心数据模型与类型定义。

该模块只放“纯数据结构”，避免与算法实现耦合，便于：
1) 在 FK/IK/RMP 等子模块之间复用统一类型；
2) 通过类型标注明确单位与语义（deg/rad、m/mm）；
3) 减少循环依赖。
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple

import numpy as np

# 关节角输入语义：
# - machine_deg: 机器角（通常与控制器内部定义一致）
# - pendant_deg: 示教器显示角
# - xml_delta_dir: (joint - offset) * direction 后再转弧度
# - raw_deg: 仅做 deg->rad（可叠加 joint_direction）
JointAngleMode = Literal["machine_deg", "pendant_deg", "xml_delta_dir", "raw_deg"]

# standard DH 单行定义：(theta_offset, d, a, alpha)
DhRow = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TeachPendantPose:
    """示教器标定表中的单行记录。

    语义约定：
    - joint_deg 为示教器/控制器约定的关节角（单位 deg）。
    - position_mm 与 orientation_rxyz_deg 为基座系下法兰报告位姿。
    """

    # 标定点序号（通常对应现场记录中的“时刻 1~N”）。
    index: int
    # 6 轴关节角，单位 deg。
    joint_deg: Tuple[float, float, float, float, float, float]
    # 法兰平移，单位 mm（基座系）。
    position_mm: Tuple[float, float, float]
    # 法兰姿态 RX/RY/RZ，单位 deg（欧拉定义由调用方决定）。
    orientation_rxyz_deg: Tuple[float, float, float]


@dataclass
class KinematicConfig:
    """运动学求解所需的最小配置集。

    注意：该结构不包含轨迹规划器状态，仅包含几何和角度映射相关参数。
    """

    # XML 或常量中的关节零位偏置，单位 deg。
    joint_offset_deg: np.ndarray
    # 关节正负方向系数（通常为 ±1）。
    joint_dir: np.ndarray
    # 机械臂类型枚举：当前实现仅支持 1（六轴偏置腕）。
    manipulator_type: int
    # 基坐标系齐次变换（世界/底座对齐结果），4x4。
    T_base: np.ndarray
    # DH 参数表（6 行）。
    dh_table: Tuple[DhRow, ...]
    # 连杆长度，单位 m，便于数值计算。
    link_len_m: np.ndarray


class IkResult(NamedTuple):
    """逆运动学返回明细（return_details=True 时使用）。"""

    # 求解得到的 6 轴角度，单位 deg。
    joint_deg: np.ndarray
    # 是否通过位置/姿态阈值检查。
    success: bool
    # `scipy.optimize.least_squares` 的收敛信息。
    message: str
    # 残差函数评估次数（不是迭代次数）。
    nfev: int
    # 最终 6 维残差向量的 L2 范数。
    residual_norm: float
