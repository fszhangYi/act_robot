from __future__ import annotations

"""正运动学接口层。

该模块负责：
1) 不同关节角语义（machine/pendant/xml/raw）的统一归一；
2) 组合基础 DH 运算得到法兰/工具位姿；
3) 提供常用位置提取接口（mm）。
"""

from typing import Optional, Sequence, Union

import numpy as np

from .config import get_kinematic_config
from .dh import fk_standard_dh
from .model import JointAngleMode, KinematicConfig


def joint_q_rad(
    jdeg: np.ndarray,
    joint_angle_mode: JointAngleMode,
    cfg: KinematicConfig,
    joint_direction: Optional[np.ndarray],
) -> np.ndarray:
    """将输入的关节角度（度）统一转换为标准 FK 所使用的弧度向量。

    不同来源的关节角度（示教器、机器人口、XML 参数、原始编码器）具有不同的
    零点偏移和方向约定。本函数根据 `joint_angle_mode` 选择相应的转换规则，
    输出可直接用于 DH 正运动学计算的关节弧度值。

    参数：
        jdeg: 形状 (6,) 的数组，表示六轴关节角度（单位：度）。
        joint_angle_mode: 枚举值，指定输入角度的语义，可选：
            - "machine_deg"   : 机器人口（控制器）直接读出的角度，已包含厂家零点偏移。
            - "pendant_deg"   : 示教器显示的角度，其方向约定通常与控制器相反。
            - "xml_delta_dir" : 基于 XML 模型定义的角度，公式：q_kin = (q_input - offset) * dir。
            - "raw_deg"       : 原始编码器角度，未经任何偏移或方向修正。
        cfg: 运动学配置对象，包含关节偏移 `joint_offset_deg` 和预设方向 `joint_dir`。
        joint_direction: 可选，6 维方向系数（+1 或 -1）。若提供则覆盖 cfg 中的默认方向值，
                        用于特殊情况下的方向重载（如单轴调试）。

    返回：
        np.ndarray: 形状 (6,) 的弧度值数组，可直接传给 `fk_standard_dh`。

    异常：
        ValueError: 如果 `joint_angle_mode` 不是支持的类型。

    注意：
        - 对于 "machine_deg" 和 "raw_deg"，如果 `joint_direction` 不为 None，则会在 rad 转换后
          乘上该方向系数（通常用于修正电机装配方向）。
        - "pendant_deg" 模式默认方向为 -1（因为示教器通常显示与控制器相反符号），
          除非显式传入 `joint_direction` 覆盖。
        - "xml_delta_dir" 模式对应标准机器人描述文件中的语义：`(q - offset) * dir`，
          其中 offset 为厂家零点偏移，dir 为 ±1。
    """
    jdeg = np.asarray(jdeg, dtype=float).reshape(6)

    if joint_angle_mode == "machine_deg":
        # 机器入口数据：直接转换为弧度，再乘上可选方向系数
        q = np.deg2rad(jdeg)
        if joint_direction is not None:
            q = q * np.asarray(joint_direction, dtype=float).reshape(6)
        return q

    if joint_angle_mode == "pendant_deg":
        # 示教器数据：先乘上方向（默认 -1），再转弧度
        dirs = (
            np.asarray(joint_direction, dtype=float).reshape(6)
            if joint_direction is not None
            else np.full(6, -1.0, dtype=float)
        )
        return np.deg2rad(jdeg * dirs)

    if joint_angle_mode == "xml_delta_dir":
        # XML 模型语义：(原始角度 - 偏移) * 方向
        dirs = (
            np.asarray(joint_direction, dtype=float).reshape(6)
            if joint_direction is not None
            else cfg.joint_dir
        )
        return np.deg2rad((jdeg - cfg.joint_offset_deg) * dirs)

    if joint_angle_mode == "raw_deg":
        # 原始编码器角度：直接转弧度，可选方向修正
        q = np.deg2rad(jdeg)
        if joint_direction is not None:
            q = q * np.asarray(joint_direction, dtype=float).reshape(6)
        return q

    raise ValueError(joint_angle_mode)


def fk_flange(
    joint_deg: Union[np.ndarray, Sequence[float]],
    *,
    T_base: Optional[np.ndarray] = None,
    joint_direction: Optional[np.ndarray] = None,
    joint_angle_mode: JointAngleMode = "machine_deg",
    kin_cfg: Optional[KinematicConfig] = None,
) -> np.ndarray:
    """计算机器人基座到法兰盘（未安装工具）的齐次变换矩阵（4x4，单位：米）。

    该函数是外部调用正向运动学的主要接口。它根据关节角度（度）和指定的语义模式，
    计算出法兰盘中心点在机器人基坐标系下的位姿。若需要包含工具变换，请使用 `fk_tool`。

    参数：
        joint_deg: 长度为 6 的关节角度序列（单位：度），顺序为 J1..J6。
        T_base:    可选，4x4 齐次矩阵，表示用户自定义的基座标变换（默认使用 cfg.T_base）。
        joint_direction: 可选，6 维方向系数，覆盖默认方向（用于特殊修正）。
        joint_angle_mode: 输入角度语义模式，见 `joint_q_rad`。
        kin_cfg:   可选，运动学配置对象。若未提供，将使用全局缓存的配置。

    返回：
        np.ndarray: 4x4 齐次变换矩阵，表示法兰盘坐标系在基坐标系下的位姿。

    异常：
        ValueError: 当 `cfg.manipulator_type != 1` 时抛出，因为当前实现仅支持六轴偏置腕。

    注意：
        - 该函数不会自动应用工具偏移（TCP）。如需 TCP 位姿，请调用 `fk_tool`。
        - 返回的平移部分单位为米（m），与 DH 参数表中的单位一致。
    """
    cfg = kin_cfg if kin_cfg is not None else get_kinematic_config()
    if cfg.manipulator_type != 1:
        raise ValueError(f"Unsupported ManipulatorType={cfg.manipulator_type} (only type 1 is supported).")

    # 将输入的关节角（度）转换为标准 DH 需要的弧度向量
    q = joint_q_rad(np.asarray(joint_deg, dtype=float), joint_angle_mode, cfg, joint_direction)

    # 基座标变换：优先使用传入的 T_base，否则使用配置中的 T_base
    T0 = cfg.T_base if T_base is None else np.asarray(T_base, dtype=float).copy()

    # 调用底层标准 DH 正向运动学
    return fk_standard_dh(q, T0, cfg.dh_table)


def fk_tool(joint_deg: Union[np.ndarray, Sequence[float]], T_flange_tool: np.ndarray, **kwargs) -> np.ndarray:
    """计算机器人基座到工具中心点（TCP）的齐次变换矩阵（4x4，单位：米）。

    该函数在法兰位姿基础上，右乘工具坐标系到法兰盘的变换矩阵 `T_flange_tool`，
    得到工具末端执行器的实际位姿。

    参数：
        joint_deg: 长度为 6 的关节角度（度），与 `fk_flange` 定义一致。
        T_flange_tool: 4x4 齐次矩阵，描述工具坐标系相对于法兰盘坐标系的位姿。
        **kwargs: 传递给 `fk_flange` 的其他关键字参数，例如：
            - T_base
            - joint_direction
            - joint_angle_mode
            - kin_cfg

    返回：
        np.ndarray: 4x4 齐次变换矩阵，表示 TCP 在机器人基坐标系下的位姿。
    """
    return fk_flange(joint_deg, **kwargs) @ T_flange_tool


def fk_position_mm(joint_deg: Union[np.ndarray, Sequence[float]], **kwargs) -> np.ndarray:
    """仅返回法兰盘中心在机器人基坐标系下的三维平移向量（单位：毫米）。

    该函数是 `fk_flange` 的轻量级包装，仅提取位置分量并转换为毫米。
    常用于高频实时监控、可视化或简单位置比较，避免处理完整的 4x4 矩阵。

    参数：
        joint_deg: 长度为 6 的关节角度（度）。
        **kwargs: 传递给 `fk_flange` 的其他参数（如 T_base, joint_angle_mode 等）。

    返回：
        np.ndarray: 形状 (3,) 的数组，表示法兰位置 [x, y, z]，单位毫米。
    
    注意：
        若只需要位置信息，使用此函数比 `fk_flange` 更高效，因为避免了完整的矩阵乘法结果拷贝。
        内部实现直接调用 `fk_flange` 再提取平移分量，若性能敏感可考虑提供专用优化路径。
    """
    return fk_flange(joint_deg, **kwargs)[:3, 3] * 1000.0