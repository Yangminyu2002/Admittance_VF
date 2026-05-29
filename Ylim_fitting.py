import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # 新增：用于路径拼接和获取
import VF_lib

# ==========================================
# 1. 数据读取与路径设置
# ==========================================
filename = r"D:\PSCAD_Learning\AIM_results\testbed_8800MW_Type4&TG\Type4_X=0.5_vw=8_ip\det_Ylim_results.txt"
# 自动获取保存目录（也就是读取文件所在的文件夹）
save_dir = os.path.dirname(filename)

df = pd.read_csv(filename, sep=r'\s+', nrows=120)
f = df['Frequency_Hz'].values
s = 1j * 2 * np.pi * f
det_Z = df['Real_Part'].values + 1j * df['Imag_Part'].values

# ==========================================
# 2. 矢量拟合 (Vector Fitting)
# ==========================================
n_order = 6 # 设定拟合阶数
poles, zeros, det_Z_fit, residues, d_const = VF_lib.vector_fitting_zeros(s, det_Z, n_order)
#poles, zeros, det_Z_fit, residues, d_const = VF_lib.vector_fitting_zeros_conjugate(s, det_Z, n_order)

# ==========================================
# 3. 结果整理与打印
# ==========================================
modes = pd.DataFrame({
    'Real_Sigma': zeros.real,
    'Imag_Omega': zeros.imag,
    'Freq_Hz': zeros.imag / (2 * np.pi),
    'Damping_Ratio_%': -zeros.real / np.abs(zeros) * 100
}).sort_values('Freq_Hz')

def print_analytical_expression(poles, residues, d, var_name='s'):
    """生成并打印拟合函数的解析表达式"""
    expr_parts = [f"{d.real:.4e} + {d.imag:.4e}j" if d.imag != 0 else f"{d.real:.4e}"]
    for i in range(len(poles)):
        p = poles[i]
        r = residues[i]
        r_str = f"({r.real:.4e} + {r.imag:.4e}j)"
        p_str = f"({p.real:.4e} + {p.imag:.4e}j)"
        expr_parts.append(f"{r_str} / ({var_name} - {p_str})")

    full_expr = " + ".join(expr_parts)
    print("\n--- det_Z_fit 解析表达式 ---")
    print(f"f({var_name}) = {full_expr}")
    return full_expr

# 调用并获取表达式字符串
analytical_expr = print_analytical_expression(poles, residues, d_const)

print("\n--- 振荡模态表 ---")
print(modes)

# ==========================================
# 4. 保存文本数据 (txt)
# ==========================================
txt_save_path = os.path.join(save_dir, 'VF_fit_TransferFunction_and_Modes.txt')
with open(txt_save_path, 'w', encoding='utf-8') as f_out:
    f_out.write("====== Vector Fitting 解析表达式 ======\n\n")
    f_out.write(f"f(s) = {analytical_expr}\n\n")
    f_out.write("====== 振荡模态表 (基于 Zeros) ======\n\n")
    f_out.write(modes.to_string(index=False))

print(f"\n[OK] 表达式和模态表已成功保存至: {txt_save_path}")

# ==========================================
# 5. 绘图与保存图片 (png)
# ==========================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(f, np.abs(det_Z), 'b', label='Original $|det(Y)|$')
plt.plot(f, np.abs(det_Z_fit), 'r--', label='VF Fit')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f, np.angle(det_Z), 'b', label='Original Phase')
plt.plot(f, np.angle(det_Z_fit), 'r--', label='VF Fit')
plt.ylabel('Phase (rad)')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# 在 plt.show() 之前保存图片
png_save_path = os.path.join(save_dir, 'VF_fit_plot.png')
plt.savefig(png_save_path, dpi=300, bbox_inches='tight')
print(f"[OK] 拟合对比图已成功保存至: {png_save_path}\n")

plt.show()