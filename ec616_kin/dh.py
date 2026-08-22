from __future__ import annotations

"""DH 链式运算基础模块。

集中承载：
- 从连杆长度构造标准 DH 表；
- 单节 DH 齐次变换；
- 全链 FK 与中间连杆位姿累积。
"""

import numpy as np
from typing import List, Tuple

from .constants import PI2
from .model import DhRow


def dh_table_from_link_lengths_m(L: np.ndarray) -> Tuple[DhRow, ...]:
    """根据 6 个连杆参数（米）生成标准 DH 参数表。

    标准 DH 建模使用四个参数 (θ, d, a, α) 描述相邻连杆间的变换关系。
    对于常见的六轴偏置腕工业机械臂，DH 参数表中：
        - theta 为绕 Z_{i-1} 轴的关节变量（偏移量），此处为 0 表示所有关节偏移已由 `JOINT_OFFSET_DEG` 单独处理；
        - d     为沿 Z_{i-1} 轴的连杆偏距（单位：米）；
        - a     为沿 X_i 轴的连杆长度（单位：米）；
        - alpha 为绕 X_i 轴的扭转角（单位：弧度），通常为 0 或 ±π/2。
    
    本函数输入的 L 长度为 6，依次对应六个连杆的几何尺寸：
        L[0] = d1   (基座到第一关节的垂直距离)
        L[1] = a2   (关节2与关节3之间的水平偏移)
        L[2] = a3   (关节3与关节4之间的水平偏移)
        L[3] = d4   (关节3与关节4之间的垂直偏置)
        L[4] = d5   (关节4与关节5之间的垂直偏置)
        L[5] = d6   (关节5与工具安装法兰的垂直偏置)

    参数：
        L: 长度为 6 的一维数组或列表，元素为连杆长度（单位：米）。
    
    返回：
        tuple[DhRow, ...]: 包含 6 个 DhRow 命名元组的序列，每个元组对应一个 DH 连杆节。
                          每个 DhRow 为 (theta_offset, d, a, alpha)，其中 theta_offset 固定为 0.0。
    
    注意：
        标准 DH 参数中的 theta 变量等于关节角度加上该节固定的 theta_offset。
        此处将 theta_offset 设为 0 表示所有角度增量完全由输入的关节角度提供，
        实际使用时应将 `JOINT_OFFSET_DEG` 转换为弧度后加到关节输入上。
    """
    L = np.asarray(L, dtype=float).reshape(6)
    l1, l2, l3, l4, l5, l6 = (float(x) for x in L)
    # 标准 DH 表：
    #   关节 i: (theta_offset, d, a, alpha)
    # 对于六轴偏置腕结构，通常第 4、5 关节存在 -90° 扭转角（-PI2）
    return (
        (0.0, l1, 0.0, -PI2),   # 关节1:  d = l1, α = -90°
        (0.0, 0.0, l2, 0.0),    # 关节2:  a = l2, α = 0°
        (0.0, 0.0, l3, 0.0),    # 关节3:  a = l3, α = 0°
        (0.0, l4, 0.0, -PI2),   # 关节4:  d = l4, α = -90°
        (0.0, l5, 0.0, -PI2),   # 关节5:  d = l5, α = -90°
        (0.0, l6, 0.0, 0.0),    # 关节6:  d = l6, α = 0°
    )


def standard_dh_link(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """计算标准 DH 模型下单个连杆的齐次变换矩阵。

    标准 DH 变换顺序（从连杆 i-1 坐标系到连杆 i 坐标系）：
        1. 绕 Z_{i-1} 轴旋转 theta 角度（关节变量）
        2. 沿 Z_{i-1} 轴平移 d 距离
        3. 沿 X_i 轴平移 a 距离
        4. 绕 X_i 轴旋转 alpha 角度

    组合变换矩阵为：
        T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)

    参数：
        theta: 关节转角（弧度），即绕 Z_{i-1} 轴旋转的角度。
        d:     连杆偏距（米），沿 Z_{i-1} 轴的平移量。
        a:     连杆长度（米），沿 X_i 轴的平移量。
        alpha: 连杆扭转角（弧度），绕 X_i 轴的旋转角度。

    返回：
        np.ndarray: 4x4 齐次变换矩阵，将点在连杆 i-1 坐标系下的坐标变换到连杆 i 坐标系。
    
    数学表达式（标准形式）：
        [   cosθ    -sinθ·cosα    sinθ·sinα    a·cosθ   ]
        [   sinθ     cosθ·cosα   -cosθ·sinα    a·sinθ   ]
        [     0         sinα         cosα         d      ]
        [     0          0            0          1      ]
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def fk_standard_dh(q_rad: np.ndarray, T_base: np.ndarray, dh_table: Tuple[DhRow, ...]) -> np.ndarray:
    """执行六轴机械臂的标准 DH 正向运动学（FK）。

    根据给定的关节角度（弧度），遍历 DH 参数表，累积各个连杆变换，最终得到
    末端执行器（工具安装法兰盘）在机器人基坐标系下的齐次变换矩阵。

    正向运动学公式：
        T_final = T_base * T_1(q1) * T_2(q2) * ... * T_6(q6)
    其中每个 T_i(qi) 使用 standard_dh_link 计算，theta = theta_offset_i + q_i。

    参数：
        q_rad:    6 维数组，表示各关节的旋转角度（单位：弧度）。顺序为 J1...J6。
        T_base:   4x4 齐次矩阵，将机器人基坐标系变换到实际安装后的用户坐标系
                  （通常为常量，包含基座标旋转和平移）。
        dh_table: 元组，包含 6 个 DhRow 条目。每个 DhRow 为 (theta_offset, d, a, alpha)，
                  其中 theta_offset 是关节零点偏移（弧度），应与 JOIN_OFFSET_DEG 转换后一致。
    
    返回：
        np.ndarray: 4x4 齐次变换矩阵，表示末端执行器在基坐标系（T_base 定义）下的位姿。
    
    注意：
        输入的 q_rad 应为已经过关节方向修正的值，即：
            q_rad = joint_dir * (encoder_reading - zero_offset)
        本函数不加额外修正，直接使用 q_rad 作为关节变量。
    """
    T = np.asarray(T_base, dtype=float).copy()
    q_rad = np.asarray(q_rad, dtype=float).reshape(6)
    for i, (th_off, d, a, al) in enumerate(dh_table):
        # theta = 该关节的固定偏移（通常为 0） + 实际关节角度
        T = T @ standard_dh_link(th_off + float(q_rad[i]), d, a, al)
    return T


def fk_link_poses_accumulated(
    q_rad: np.ndarray,
    T_base: np.ndarray,
    dh_table: Tuple[DhRow, ...],
) -> List[np.ndarray]:
    """计算各连杆（含基座）在机器人基坐标系下的累积位姿。

    该函数不仅返回末端位姿，还返回每个关节输出坐标系（即每个连杆 i 的坐标系）
    在基坐标系下的表示。通常用于可视化、碰撞检测或运动学分解。

    累积变换序列：
        T_0 = T_base                # 基座（用户坐标系）
        T_1 = T_0 * T_joint1        # 关节1后的位姿
        T_2 = T_1 * T_joint2
        ...
        T_6 = T_5 * T_joint6        # 末端法兰盘位姿

    参数：
        q_rad:    6 维数组，关节角度（弧度）。
        T_base:   4x4 矩阵，基坐标系到用户坐标系的变换。
        dh_table: DH 参数表（与 fk_standard_dh 一致）。

    返回：
        list[np.ndarray]: 长度为 7 的列表，元素依次为：
            [T_base, T_after_J1, T_after_J2, ..., T_after_J6]
        每个元素均为 4x4 齐次矩阵。
    """
    T = np.asarray(T_base, dtype=float).copy()
    out: List[np.ndarray] = [T.copy()]      # T_base 作为初始位姿
    qv = np.asarray(q_rad, dtype=float).reshape(6)
    for i, (th_off, d, a, al) in enumerate(dh_table):
        T = T @ standard_dh_link(th_off + float(qv[i]), d, a, al)
        out.append(T.copy())                # 保存当前连杆位姿
    return out