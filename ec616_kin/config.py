from __future__ import annotations

"""运动学配置构建与缓存。

职责：
- 将常量层参数转换为数值求解直接可用的 `KinematicConfig`；
- 提供惰性缓存，避免重复构造造成开销与状态不一致。
"""

import numpy as np
from typing import Optional

from .constants import (
    BASE_CS_X,
    BASE_CS_Z,
    JOINT_DIRECTION,
    JOINT_OFFSET_DEG,
    LINK_LENGTH_MM,
    MANIPULATOR_TYPE,
    _AXIS_VEC,
)
from .dh import dh_table_from_link_lengths_m
from .model import KinematicConfig
from .teach import teach_base_rotation_from_install_deg


def cs_to_rotation(base_dir_x: int, base_dir_z: int) -> np.ndarray:
    """根据控制器坐标系轴向枚举值计算正交旋转矩阵。

    在机器人运动学标定或配置加载时，需要将用户定义的基坐标系轴向（通常为X, Z轴的方向）
    转换为一个3×3的旋转矩阵，该矩阵描述了从控制器定义坐标系到标准机器人基坐标系的旋转。

    参数：
        base_dir_x: 整数枚举，表示基坐标系X轴的正方向。
                   取值来自 `_AXIS_VEC` 的键，例如 0~6 对应 X, Y, Z 及其负方向。
        base_dir_z: 整数枚举，表示基坐标系Z轴的正方向。

    返回：
        3×3 正交旋转矩阵，满足右手系且各列单位长度。
        如果X轴与Z轴几乎共线（点积绝对值 > 0.98），则返回单位阵作为保守降级处理，
        防止后续数值计算出现奇异。

    算法步骤：
        1. 根据枚举值从预定义的轴向向量字典 `_AXIS_VEC` 中获取对应的单位向量。
        2. 将X轴和Z轴归一化（确保数值稳定）。
        3. 若X与Z的点积绝对值 > 0.98，说明两轴接近平行（非正交基），直接返回单位阵。
        4. 使用Z轴与X轴叉乘得到临时Y轴：y_temp = cross(z, x)，再归一化。
        5. 用Y轴与Z轴叉乘重新计算X轴：x_new = cross(y_temp, z)，以保证三轴严格正交且构成右手系。
        6. 返回列堆叠的旋转矩阵 [x_new, y_temp, z]。
    """
    # 根据枚举键获取对应的轴向单位向量，若键不存在则退化为默认Z轴方向（键1对应Z+）
    x = np.asarray(_AXIS_VEC.get(base_dir_x, _AXIS_VEC[6]), dtype=float)
    z = np.asarray(_AXIS_VEC.get(base_dir_z, _AXIS_VEC[1]), dtype=float)

    # 归一化，避免因常量定义中的非单位向量造成误差
    x /= np.linalg.norm(x)
    z /= np.linalg.norm(z)

    # 检查两轴是否几乎共线（不满足右手系正交要求）
    if abs(np.dot(x, z)) > 0.98:
        # 降级：返回单位矩阵，后续可以依赖其他基座标定补偿
        return np.eye(3)

    # 构造正交化右手系：先用z×x得到y方向，再反向修正x以保证完美正交
    y = np.cross(z, x)          # 叉积顺序保证 y = z × x
    y /= np.linalg.norm(y)
    x = np.cross(y, z)          # 重新计算 x = y × z，确保 x, y, z 两两正交
    x /= np.linalg.norm(x)

    # 堆叠列向量得到旋转矩阵（X, Y, Z 列）
    return np.column_stack([x, y, z])


def build_kinematic_config() -> KinematicConfig:
    """构建完整的 `KinematicConfig` 对象，不使用缓存。

    该函数将常量定义（连杆长度、关节偏移、关节方向、基坐标系旋转等）组装成一个
    运动学配置对象，供正逆运动学求解及动力学计算使用。

    返回：
        KinematicConfig: 包含以下内容的数据类实例：
            - joint_offset_deg: 关节零位偏移（度），用于修正编码器零点与理论零位的偏差。
            - joint_dir: 关节旋转方向（+1 或 -1），补偿关节装配方向与模型定义的差异。
            - manipulator_type: 机械臂类型标识（如 '6dof' 等）。
            - T_base: 基坐标系变换矩阵（4×4），将末端位姿从机器人安装面转换到用户定义的基座标系。
            - dh_table: DH 参数表（4×N 矩阵），包含 a, α, d, θ 的标准参数。
            - link_len_m: 各连杆长度（米），复制自常量定义，便于外部访问。
    """
    # 将连杆长度从毫米转换为米
    link_m = np.array(LINK_LENGTH_MM, dtype=float) / 1000.0

    # 计算机器人基坐标系的旋转矩阵：
    # 1. 根据控制器轴向枚举得到从控制器定义到机器人本体系的旋转 `cs_to_rotation`
    # 2. 再乘以基于安装角度的标定旋转 `teach_base_rotation_from_install_deg()`。
    #    后者通常通过手眼标定或人工示教获得。
    R = cs_to_rotation(BASE_CS_X, BASE_CS_Z) @ teach_base_rotation_from_install_deg()

    # 构建基坐标系齐次变换矩阵（平移部分通常为0，因为基座原点与安装面重合）
    T_base = np.eye(4)
    T_base[:3, :3] = R

    # 组装并返回运动学配置对象
    return KinematicConfig(
        joint_offset_deg=np.array(JOINT_OFFSET_DEG, dtype=float),
        joint_dir=np.array(JOINT_DIRECTION, dtype=float),
        manipulator_type=MANIPULATOR_TYPE,
        T_base=T_base,
        dh_table=dh_table_from_link_lengths_m(link_m),   # 根据连杆长度列表生成标准DH参数
        link_len_m=link_m.copy(),                       # 深拷贝，防止外部修改
    )


# 全局缓存变量：存储单例运动学配置，避免重复构建
_CFG: Optional[KinematicConfig] = None


def get_kinematic_config() -> KinematicConfig:
    """获取全局缓存运动学配置，惰性初始化。

    该函数是外部模块获取运动学配置的推荐入口，它会自动在首次调用时构建配置，
    后续调用直接返回同一实例，避免了重复构造带来的性能开销和潜在状态不一致。

    若需要强制重新构建（例如修改了常量定义），则应调用 `clear_kinematic_config_cache()`
    清除缓存，然后再次调用本函数。

    返回：
        KinematicConfig: 当前有效的运动学配置对象。
    """
    global _CFG
    if _CFG is None:
        _CFG = build_kinematic_config()
    return _CFG


def clear_kinematic_config_cache() -> None:
    """清空全局运动学配置缓存。

    当常量模块中的参数（如连杆长度、关节方向或基座标旋转）发生动态变化时，
    调用此函数使缓存失效，确保下次 `get_kinematic_config()` 会重新构建最新的配置。

    注意：在高实时性场景中，应避免频繁清空缓存导致重复构建开销。
    """
    global _CFG
    _CFG = None