import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Vac_base = 575
S_base = 2e6
Z_base = Vac_base**2 / S_base
Y_base = 1/Z_base

def load_and_convert_to_complex_Y(filename):
    """读取文本文件并将 dB/Degree 转换为复数 (针对导纳 Y)"""
    df = pd.read_csv(filename, sep=r'\s+')

    # 提取频率
    f = df['fp'].values

    # 内部转换函数：dB -> 线性幅值，度数 -> 弧度，最后组合为复数
    def to_complex(mag_db, pha_deg):
        mag_lin = 10 ** (mag_db / 20.0)  # dB 转线性幅值
        pha_rad = np.deg2rad(pha_deg)  # 角度转弧度
        return mag_lin * np.exp(1j * pha_rad)  # 构造复数 Y = |Y| * e^(j*θ)

    # 提取 dd, dq, qd, qq 四个分量并转换为复数数组
    # 注意：这里的列名改为了 Ydd_mag, Ydd_pha 等
    Y = {
        'dd': to_complex(df['Ydd_mag'], df['Ydd_pha']),
        'dq': to_complex(df['Ydq_mag'], df['Ydq_pha']),
        'qd': to_complex(df['Yqd_mag'], df['Yqd_pha']),
        'qq': to_complex(df['Yqq_mag'], df['Yqq_pha'])
    }

    return f, Y


# ==========================================
# 1. 加载两个导纳文件数据
# ==========================================
# 请确保这里的文件名与你实际的导纳文件名一致
f, Y_s = load_and_convert_to_complex_Y(r"D:\PSCAD_Learning\AIM_results\testbed_8800MW_Type4&TG\Type4_X=0.3_ip\Measurementdata_Toolbox1\Terminal2\T2_admittance_dq_MIMO1.txt")
_, Y_g = load_and_convert_to_complex_Y(r'D:\PSCAD_Learning\AIM_results\testbed_8800MW_Type4&TG\Type4_X=0.3_ip\Measurementdata_Toolbox1\Terminal1\T1_admittance_dq_MIMO1.txt')

# ==========================================
# 2. 计算 Y_LIM = Y_s + Y_g
# ==========================================
Y_lim = {}
components = ['dd', 'dq', 'qd', 'qq']
for comp in components:
    Y_lim[comp] = Y_s[comp] + Y_g[comp]  # 频域下复数直接相加



# ==========================================
# 3. 计算 Y_LIM 的行列式 det(Y_LIM)
# ==========================================
# det(Y) = Y_dd * Y_qq - Y_dq * Y_qd
det_Y_lim = Y_lim['dd'] * Y_lim['qq'] - Y_lim['dq'] * Y_lim['qd']

# ==========================================
# 4. 将行列式复数结果转回 幅值(dB) 和 相位(度)
# ==========================================
mag_db = 20 * np.log10(np.abs(det_Y_lim))  # 线性幅值转 dB
mag = np.abs(det_Y_lim)  # 线性幅值转 dB
#mag_db = np.abs(det_Y_lim)/Y_base # 如果你想看线性幅值可以取消注释这行
pha_deg = np.rad2deg(np.angle(det_Y_lim))  # 提取相位角并转为度数

# ==========================================
# 5. 绘制 det(Y_LIM) 的 Bode 图 (线性坐标)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle(r'Bode Plot of $\det(Y_{LAM})$ (Linear Scale)', fontsize=16)

# 绘制幅频特性
ax1.plot(f, mag_db, color='g', linewidth=1.5, label=r'$|\det(Y_{LAM})|$') # 颜色换成了绿色区分
ax1.set_ylabel('Magnitude (dB)')
ax1.grid(True, which="both", ls="--", alpha=0.7)
ax1.legend(loc='upper right')
ax1.set_xlim([min(f), max(f)])

# 绘制相频特性
ax2.plot(f, pha_deg, color='orange', linewidth=1.5, label=r'Phase of $\det(Y_{LAM})$') # 颜色换成了橙色区分
ax2.set_ylabel('Phase (deg)')
ax2.set_xlabel('Frequency (Hz)')
ax2.grid(True, which="both", ls="--", alpha=0.7)
ax2.legend(loc='upper right')
ax2.set_xlim([min(f), max(f)])

plt.tight_layout()
fig.subplots_adjust(top=0.92)  # 为总标题腾出空间
plt.show()

output_df = pd.DataFrame({
    'Frequency_Hz': f,
    'Real_Part': np.real(det_Y_lim),
    'Imag_Part': np.imag(det_Y_lim),
    'Magnitude': mag,
    'Phase_deg': pha_deg
})

# 保存为以空格分隔的文本文件（不包含索引）
output_filename = r'D:\PSCAD_Learning\AIM_results\testbed_8800MW_Type4&TG\Type4_X=0.3_ip\det_Ylim_results.txt'
output_df.to_csv(output_filename, sep=' ', index=False, float_format='%.6e')

print(f"结果已成功保存至: {output_filename}")

#绘制YlAM的MIMO bode图
def plot_bode_mimo_linear(f, Y_lim):
    """
    绘制 MIMO 系统的 Bode 图，使用线性频率坐标
    """
    # 4 行 2 列的布局
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

    # 布局逻辑
    layout_map = [
        ('dd', 'Ydd(s)'), ('dq', 'Ydq(s)'),
        ('dd', 'Ydd(s)'), ('dq', 'Ydq(s)'),
        ('qd', 'Yqd(s)'), ('qq', 'Yqq(s)'),
        ('qd', 'Yqd(s)'), ('qq', 'Yqq(s)')
    ]

    for i, (key, title) in enumerate(layout_map):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # 核心改动：使用 plot() 代替 semilogx()
        if row % 2 == 0:
            # 绘制幅值
            mag_db = 20 * np.log10(np.abs(Y_lim[key]))
            ax.plot(f, mag_db, color='tab:blue', linewidth=1.5)
            ax.set_ylabel('Mag (dB)')
            ax.set_title(title, fontweight='bold')
        else:
            # 绘制相位
            pha_deg = np.rad2deg(np.angle(Y_lim[key]))
            ax.plot(f, pha_deg, color='tab:red', linewidth=1.5)
            ax.set_ylabel('Phase (deg)')

        ax.grid(True, linestyle='--', alpha=0.7)

        # 建议添加 xlim 来控制显示范围，线性坐标下这很有用
        # ax.set_xlim(0, 500)

        if row == 3:
            ax.set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


# 调用函数
plot_bode_mimo_linear(f, Y_lim)
# 1. 计算幅值 (dB)
mag_dd = 20 * np.log10(np.abs(Y_lim['dd']))

# 2. 找到最大值所在的索引
idx = np.argmax(mag_dd)

# 3. 提取对应的频率
peak_freq = f[idx]

print(f"Ydd 的谐振峰频率为: {peak_freq:.2f} Hz")
print(f"对应的谐振峰值: {mag_dd[idx]:.2f} dB")

