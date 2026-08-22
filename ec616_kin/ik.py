from __future__ import annotations

"""数值逆运动学模块（基于 SciPy least_squares）。

实现要点：
- 误差定义为 SE(3) 6 维残差：位置误差 + 加权旋转向量误差；
- 支持 LM（快、无边界）与 TRF（支持边界）两种求解策略；
- 可选施加示教软限位边界。
"""

from typing import Literal, Optional, Union

import numpy as np
from scipy.optimize import least_squares

from .config import get_kinematic_config
from .dh import fk_standard_dh
from .model import IkResult, KinematicConfig
from .teach import joints_within_teach_soft_limits, teach_joint_soft_limits_deg


def rotvec_so3(R: np.ndarray) -> np.ndarray:
    """将 SO(3) 旋转矩阵映射为旋转向量（axis-angle）。

    旋转向量 ω = θ * k，其中 θ 为旋转角度，k 为单位旋转轴。
    该函数实现了从旋转矩阵到旋转向量的数值稳定转换，
    适用于小角度和接近 π 角度的情况。

    数值策略：
    - 小角度附近（θ < 1e-9）：直接返回零向量，避免除零。
    - 一般情况：使用 Rodrigues 公式的逆映射：ω = φ * (θ / sinθ)，
      其中 φ = [R32-R23, R13-R31, R21-R12]^T。
    - 接近 π 时（sinθ 很小）：改用对称矩阵特征分解估计旋转轴，
      从 R + R^T 的最大特征值对应的特征向量中提取轴方向，乘以 θ 得到 ω。
    
    参数：
        R: 3x3 旋转矩阵，应满足正交且行列式为 +1。

    返回：
        np.ndarray: 形状 (3,) 的旋转向量（轴角表示），模长等于旋转角度（弧度）。

    注意：
        当角度接近 π 时，旋转轴存在两种可能（方向相反），但该函数返回的向量模长
        在 (π - ε, π] 范围内，不影响后续优化残差的梯度。
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    # 计算旋转角 cosθ = (trace(R) - 1) / 2，并裁剪到有效范围[-1,1]
    cos_t = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    # 构造旋转向量原始分量 φ = [R32 - R23, R13 - R31, R21 - R12]
    phi = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
    
    if theta < 1e-9:
        # 零旋转，直接返回零向量
        return np.zeros(3)
    
    st = np.sin(theta)
    if abs(st) > 1e-4:
        # 常规情况：θ / sinθ 数值稳定
        return phi * (theta / st)
    
    # 当 θ 接近 π (sinθ≈0) 时，采用特征分解法求解旋转轴
    # 利用 R + R^T = 2 * (I - [k]×^2) + 2 k k^T，最大特征值对应的特征向量即为 k
    w, V = np.linalg.eigh(R + R.T)
    k = V[:, int(np.argmax(w))]          # 最大特征值对应的特征向量
    k = k / (np.linalg.norm(k) + 1e-12)  # 归一化
    return k * theta


def pose_error_se3(T_cur: np.ndarray, T_des: np.ndarray, ori_weight: float) -> np.ndarray:
    """构造逆运动学求解的目标残差向量：6维 [位置误差, 加权旋转误差]。

    参数：
        T_cur: 当前法兰位姿（4x4 齐次矩阵），由当前关节角通过正运动学计算得到。
        T_des: 期望的法兰位姿（4x4 齐次矩阵）。
        ori_weight: 姿态误差的权重系数（无量纲）。
                   因为位置误差的单位是米，旋转误差的单位是弧度，
                   两者量纲不同，通过该系数平衡两部分贡献，使优化更稳定。

    返回：
        np.ndarray: 长度为 6 的残差向量：
            e_p = P_des - P_cur           (平移部分，单位米)
            e_r = rotvec(R_des * R_cur^T) * ori_weight   (加权旋转向量)
    
    注意：
        姿态误差定义为相对旋转矩阵 R_rel = R_des * R_cur^T，
        将其转换为旋转向量后，其模长即为所需旋转角（弧度）。
    """
    # 位置误差：期望位置 - 当前位置
    e_p = T_des[:3, 3] - T_cur[:3, 3]
    # 相对旋转矩阵：当前坐标系变换到期望坐标系的旋转
    R_err = T_des[:3, :3] @ T_cur[:3, :3].T
    # 将相对旋转转换为旋转向量并加权
    e_r = rotvec_so3(R_err) * ori_weight
    return np.concatenate([e_p, e_r])


def ik_flange(
    T_desired: np.ndarray,
    q_seed_deg: np.ndarray,
    *,
    T_base: Optional[np.ndarray] = None,
    return_details: bool = False,
    max_nfev: int = 400,
    position_tolerance_m: float = 1e-5,
    orientation_tolerance_rad: float = 5e-4,
    ori_weight: float = 0.3,
    lm_method: Literal["lm", "trf"] = "lm",
    enforce_teach_soft_limits: bool = False,
    kin_cfg: Optional[KinematicConfig] = None,
) -> Union[np.ndarray, IkResult]:
    """求解法兰逆运动学（数值法），返回关节角度（度）或包含详细信息的 IK 结果对象。

    该函数利用 scipy.optimize.least_squares 最小化 pose_error_se3 的 2-范数，
    找到一组关节角度使得法兰位姿逼近目标位姿。支持两种求解策略：
        - "lm" : Levenberg-Marquardt 算法（无边界约束），收敛快，适合无障碍场景。
        - "trf": Trust Region Reflective 算法（支持边界约束），可结合软限位使用。

    参数：
        T_desired: 目标法兰位姿（4x4 齐次矩阵，单位：米）。
        q_seed_deg: 初始猜测关节角度（度，长度为 6）。数值 IK 高度依赖初值，
                    建议采用最近邻解或上次求解结果。
        T_base: 可选，基坐标系变换矩阵（默认使用运动学配置中的 T_base）。
        return_details: 若为 True，返回 IkResult 结构体（包含成功标志、迭代次数、误差等）；
                        若为 False（默认），成功时返回关节角（度），失败时抛出 RuntimeError。
        max_nfev: 最大函数评估次数（默认 400）。
        position_tolerance_m: 位置误差容忍度（米，默认 1e-5，即 0.01 mm）。
        orientation_tolerance_rad: 姿态误差容忍度（弧度，默认 5e-4 ≈ 0.0286°）。
        ori_weight: 姿态残差加权系数（默认 0.3）。用于平衡位置和姿态的量纲差异。
        lm_method: 选择优化器，"lm" 或 "trf"。若 enforce_teach_soft_limits=True，
                   则自动切换为 "trf" 以支持边界。
        enforce_teach_soft_limits: 是否强制执行示教软限位（关节角度边界）。
                                   若为 True，优化将在给定边界内搜索，且最终解会检查是否超限。
        kin_cfg: 运动学配置对象，若为 None 则使用全局缓存配置。

    返回：
        - 若 return_details=False 且求解成功：返回 np.ndarray (6,) 关节角度（度）。
        - 若 return_details=True：返回 IkResult 命名元组（包含 joint_deg, success, message, nfev, residual_norm）。
        - 若求解失败且 return_details=False：抛出 RuntimeError，附带详细误差信息。

    注意：
        - 当 enforce_teach_soft_limits=True 时，初始猜测会被 clip 到边界内，避免求解器初始就越界。
        - 姿态误差收敛判据使用加权后的范数： ||e_rot|| < ori_weight * orientation_tolerance_rad。
        - 即使求解器返回成功，但最终解超出软限位时（由于近似误差），也会标记为失败。
    """
    # 获取运动学配置
    cfg = kin_cfg if kin_cfg is not None else get_kinematic_config()
    if cfg.manipulator_type != 1:
        raise ValueError(f"Unsupported ManipulatorType={cfg.manipulator_type} (only type 1 is supported).")
    
    # 确定基座标变换矩阵
    T0 = cfg.T_base if T_base is None else np.asarray(T_base, dtype=float).copy()
    # 初始猜测：度 -> 弧度，并重塑为 (6,)
    q0 = np.deg2rad(np.asarray(q_seed_deg, dtype=float).reshape(6))
    T_des = np.asarray(T_desired, dtype=float).reshape(4, 4)

    # 获取软限位边界（度 -> 弧度）
    lo_deg, hi_deg = teach_joint_soft_limits_deg()
    q_lo = np.deg2rad(lo_deg)
    q_hi = np.deg2rad(hi_deg)
    
    if enforce_teach_soft_limits:
        # 将初始猜测限制在可行域内，避免 TRF 在越界初始位置产生数值异常
        q0 = np.clip(q0, q_lo, q_hi)

    # 定义残差函数：输入为弧度关节角，输出为 6 维残差向量
    def residual(q_rad: np.ndarray) -> np.ndarray:
        T_cur = fk_standard_dh(q_rad, T0, cfg.dh_table)
        return pose_error_se3(T_cur, T_des, ori_weight)

    # 构造 least_squares 调用参数
    kwargs: dict = {
        "fun": residual,
        "x0": q0,
        "max_nfev": int(max_nfev),
        "ftol": 1e-12,   # 函数值（残差平方和）的容忍变化
        "xtol": 1e-12,   # 自变量的容忍变化
        "gtol": 1e-12,   # 梯度的容忍范数
    }
    
    # 选择求解器：若需要边界约束或显式指定 trf，则使用 trust-region reflective 方法
    use_trf = lm_method == "trf" or enforce_teach_soft_limits
    if use_trf:
        kwargs["method"] = "trf"
        kwargs["jac"] = "2-point"          # 有限差分近似雅可比（无需提供解析导数）
        if enforce_teach_soft_limits:
            kwargs["bounds"] = (q_lo, q_hi)   # 设置关节边界
    else:
        kwargs["method"] = "lm"

    # 执行优化
    res = least_squares(**kwargs)
    
    # 提取最优解（弧度并转换为度）
    q = np.asarray(res.x, dtype=float).reshape(6)
    T_chk = fk_standard_dh(q, T0, cfg.dh_table)
    e = pose_error_se3(T_chk, T_des, ori_weight)
    
    # 判断收敛性
    pos_ok = np.linalg.norm(e[:3]) < position_tolerance_m
    ori_ok = np.linalg.norm(e[3:]) < ori_weight * orientation_tolerance_rad
    ok = pos_ok and ori_ok
    q_deg = np.rad2deg(q)
    msg = str(res.message)
    
    # 若要求软限位，最终还需确认解在边界内（由于数值误差可能略微超出）
    if ok and enforce_teach_soft_limits and not joints_within_teach_soft_limits(q_deg):
        ok = False

    if return_details:
        return IkResult(
            joint_deg=q_deg,
            success=ok,
            message=msg,
            nfev=int(res.nfev),
            residual_norm=float(np.linalg.norm(e)),
        )
    
    if not ok:
        # 失败时抛出包含诊断信息的异常
        raise RuntimeError(
            f"IK pose error over tolerance (scipy: success={res.success}, {msg}); "
            f"pos_err_m={np.linalg.norm(e[:3]):.2e}, "
            f"weighted_ori_err={np.linalg.norm(e[3:]):.2e}, nfev={res.nfev}. "
            "Try another q_seed_deg, use lm_method='trf', enforce_teach_soft_limits=False, or relax tolerances.",
        )
    return q_deg