import numpy as np

def vector_fitting_zeros(s, y, n_order, iterations=10):
    """
    对 y(s) 进行矢量拟合，并提取其零点分布。
    """
    # 1. 初始极点分布 (分布在左半平面)
    f_min = np.min(s.imag / (2 * np.pi))
    f_max = np.max(s.imag / (2 * np.pi))
    init_imag = np.linspace(f_min, f_max, n_order) * (2 * np.pi)
    poles = -1.0 + 1j * init_imag

    # 2. 极点重定位迭代 (Pole Relocation)
    for i in range(iterations):
        n_pts = len(s)
        A = np.zeros((n_pts, 2 * n_order + 1), dtype=complex)
        for n in range(n_order):
            A[:, n] = 1.0 / (s - poles[n])
        A[:, n_order] = 1.0
        for n in range(n_order):
            A[:, n_order + 1 + n] = -y / (s - poles[n])

        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        c_tilde = sol[n_order + 1:]
        # 新极点是拟合函数 scaling function 的零点
        poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(n_order), c_tilde))

    # 3. 计算最终留数 r 和常数项 d
    A_final = np.zeros((len(s), n_order + 1), dtype=complex)
    for n in range(n_order):
        A_final[:, n] = 1.0 / (s - poles[n])
    A_final[:, n_order] = 1.0

    res_sol, _, _, _ = np.linalg.lstsq(A_final, y, rcond=None)
    residues = res_sol[:n_order]
    d = res_sol[n_order]
    y_fit = A_final @ res_sol

    # 4. 提取有理函数 f(s) = d + sum(r_i/(s-p_i)) 的零点
    # 零点是矩阵 (diag(Poles) - (1/d) * ones * Residues^T) 的特征值
    if np.abs(d) > 1e-10:
        zeros = np.linalg.eigvals(np.diag(poles) - (1.0 / d) * np.outer(np.ones(n_order), residues))
    else:
        zeros = np.array([])

    return poles, zeros, y_fit, residues, d

'''
def vector_fitting_zeros_new(s, y, n_order, iterations=10):
    """
    按照文档方法：拟合 1/det(Z)，利用其极点作为原函数的零点辨识结果。
    """
    # --- 文档逻辑核心：取倒数进行拟合 ---
    # 因为 det(Z) 的零点集合等价于 1/det(Z) 的极点集合
    y_inv = 1.0 / y

    # 1. 初始极点分布 (分布在左半平面)
    f_min = np.min(s.imag / (2 * np.pi))
    f_max = np.max(s.imag / (2 * np.pi))
    init_imag = np.linspace(f_min, f_max, n_order) * (2 * np.pi)
    # 文档未要求稳定性校正，但初始极点设在左半平面有助于迭代收敛
    poles_inv = -1.0 + 1j * init_imag

    # 2. 极点重定位迭代 (Pole Relocation) - 针对 1/det(Z)
    for i in range(iterations):
        n_pts = len(s)
        A = np.zeros((n_pts, 2 * n_order + 1), dtype=complex)
        for n in range(n_order):
            A[:, n] = 1.0 / (s - poles_inv[n])
        A[:, n_order] = 1.0
        for n in range(n_order):
            A[:, n_order + 1 + n] = -y_inv / (s - poles_inv[n])

        sol, _, _, _ = np.linalg.lstsq(A, y_inv, rcond=None)
        c_tilde = sol[n_order + 1:]
        # 新极点是 scaling function 的零点
        poles_inv = np.linalg.eigvals(np.diag(poles_inv) - np.outer(np.ones(n_order), c_tilde))

    # 3. 计算最终留数 r 和常数项 d (针对 1/det(Z))
    A_final = np.zeros((len(s), n_order + 1), dtype=complex)
    for n in range(n_order):
        A_final[:, n] = 1.0 / (s - poles_inv[n])
    A_final[:, n_order] = 1.0

    res_sol, _, _, _ = np.linalg.lstsq(A_final, y_inv, rcond=None)
    residues_inv = res_sol[:n_order]
    d_inv = res_sol[n_order]

    # 得到 1/det(Z) 的拟合曲线
    y_inv_fit = A_final @ res_sol
    # 原 det(Z) 的拟合曲线为倒数
    y_fit = 1.0 / y_inv_fit

    # 4. 提取零点与极点
    # 根据文档：1/det(Z) 的极点集合即为 det(Z) 的零点辨识结果
    zeros_of_det = poles_inv

    # 利用状态空间法提取 1/det(Z) 的零点，这些是原 det(Z) 的极点
    if np.abs(d_inv) > 1e-10:
        poles_of_det = np.linalg.eigvals(np.diag(poles_inv) - (1.0 / d_inv) * np.outer(np.ones(n_order), residues_inv))
    else:
        poles_of_det = np.array([])

    # 为了与原输出格式保持一致：
    # 原 poles 代表拟合函数的极点，原 zeros 代表拟合函数的零点
    # 这里的 y_fit 是针对 det(Z) 的，所以：
    return poles_of_det, zeros_of_det, y_fit, residues_inv, d_inv'''


def vector_fitting_zeros_conjugate(s, y, n_order, iterations=10):
    """
    带共轭约束的矢量拟合算法。
    确保拟合出的极点、留数为严格的共轭复数对，常数项 d 为实数。
    """
    s = np.asarray(s, dtype=complex)
    y = np.asarray(y, dtype=complex)
    # 1. 强制阶数为偶数，并初始化严格共轭的复数极点对
    n_order = n_order if n_order % 2 == 0 else n_order + 1
    f_min = np.min(s.imag / (2 * np.pi))
    f_max = np.max(s.imag / (2 * np.pi))
    init_imag = np.linspace(f_min, f_max, n_order // 2) * (2 * np.pi)

    poles = np.zeros(n_order, dtype=complex)
    poles[0::2] = -1.0 + 1j * init_imag
    poles[1::2] = -1.0 - 1j * init_imag

    n_pts = len(s)

    # 2. 极点重定位迭代
    for i in range(iterations):
        A = np.zeros((n_pts, 2 * n_order + 1), dtype=complex)

        # 核心修改：基于共轭极点对构造实数化矩阵 A
        for n in range(0, n_order, 2):
            # 拟合函数项：c1/(s-p) + c2/(s-p*) -> 转换为求纯实数 c 和 d
            A[:, n] = 1.0 / (s - poles[n]) + 1.0 / (s - poles[n + 1])
            A[:, n + 1] = 1j / (s - poles[n]) - 1j / (s - poles[n + 1])

            # 比例函数项：-y * (c3/(s-p) + c4/(s-p*))
            A[:, n_order + 1 + n] = -y * (1.0 / (s - poles[n]) + 1.0 / (s - poles[n + 1]))
            A[:, n_order + 1 + n + 1] = -y * (1j / (s - poles[n]) - 1j / (s - poles[n + 1]))

        A[:, n_order] = 1.0  # 常数项 d

        # 将复数方程组拆解为实部和虚部，强制最小二乘法解出纯实数 x
        A_real_eqs = np.vstack((A.real, A.imag))
        y_real_eqs = np.concatenate((y.real, y.imag))

        sol, _, _, _ = np.linalg.lstsq(A_real_eqs, y_real_eqs, rcond=None)

        c_tilde_real = sol[n_order + 1:]

        # 将求得的实数系数还原为复数共轭系数 c_tilde
        c_tilde = np.zeros(n_order, dtype=complex)
        for n in range(0, n_order, 2):
            c_tilde[n] = c_tilde_real[n] + 1j * c_tilde_real[n + 1]
            c_tilde[n + 1] = c_tilde_real[n] - 1j * c_tilde_real[n + 1]

        # 计算新极点
        poles = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(n_order), c_tilde))

        # 强制极点稳定 (如果跑到右半平面，则翻转回左半平面)
        poles = np.where(poles.real > 0, -poles.real + 1j * poles.imag, poles)

        # 强制清除数值误差，重新配对严格的共轭对
        # 技巧：按虚部排序，取虚部最大的那一半，强行复制为其共轭
        pos_poles = poles[np.argsort(poles.imag)][n_order // 2:]
        poles[0::2] = pos_poles
        poles[1::2] = np.conj(pos_poles)

    # 3. 计算最终留数 r 和常数项 d (使用同样的实数化逻辑)
    A_final = np.zeros((n_pts, n_order + 1), dtype=complex)
    for n in range(0, n_order, 2):
        A_final[:, n] = 1.0 / (s - poles[n]) + 1.0 / (s - poles[n + 1])
        A_final[:, n + 1] = 1j / (s - poles[n]) - 1j / (s - poles[n + 1])
    A_final[:, n_order] = 1.0

    A_final_real_eqs = np.vstack((A_final.real, A_final.imag))
    y_real_eqs = np.concatenate((y.real, y.imag))

    res_sol, _, _, _ = np.linalg.lstsq(A_final_real_eqs, y_real_eqs, rcond=None)

    d = res_sol[n_order]
    residues_real = res_sol[:n_order]

    # 还原留数
    residues = np.zeros(n_order, dtype=complex)
    for n in range(0, n_order, 2):
        residues[n] = residues_real[n] + 1j * residues_real[n + 1]
        residues[n + 1] = residues_real[n] - 1j * residues_real[n + 1]

    # 用于计算最后拟合曲线的干净矩阵
    A_curve = np.zeros((n_pts, n_order + 1), dtype=complex)
    for n in range(n_order):
        A_curve[:, n] = 1.0 / (s - poles[n])
    A_curve[:, n_order] = 1.0
    y_fit = A_curve @ np.append(residues, d)

    # 4. 提取有理函数的零点
    if np.abs(d) > 1e-10:
        zeros = np.linalg.eigvals(np.diag(poles) - (1.0 / d) * np.outer(np.ones(n_order), residues))
    else:
        zeros = np.array([])

    return poles, zeros, y_fit, residues, d