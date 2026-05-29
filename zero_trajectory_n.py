import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import os

# ==========================================
# 0. 全局设置
# ==========================================
# 移除 Times New Roman，恢复 Matplotlib 默认的无衬线字体以匹配参考图
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

data_dir = r"d:\Python\damping_modes_n"
output_dir = data_dir

n_values = [1000, 1250, 1500, 1750, 2000]
n_norm = (np.array(n_values) - min(n_values)) / (max(n_values) - min(n_values))
# 更改 colormap 为 'plasma' 以匹配参考图（深蓝 -> 紫 -> 粉 -> 橙 -> 黄）
cmap_n = plt.cm.plasma 

# ==========================================
# 1. 解析各 n 值的模态数据
# ==========================================
def parse_mode_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    marker = 'Real_Sigma  Imag_Omega    Freq_Hz  Damping_Ratio_%'
    idx = content.find(marker)
    if idx < 0:
        return []
    table_text = content[idx + len(marker):].strip()
    modes = []
    for line in table_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('===') or line.startswith('f(s)'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                sigma = float(parts[0])
                omega = float(parts[1])
                freq = float(parts[2])
                damping = float(parts[3])
                modes.append({'sigma': sigma, 'omega': omega,
                              'freq': freq, 'damping': damping})
            except ValueError:
                continue
    return modes

all_data = {}
for n in n_values:
    filepath = os.path.join(data_dir, f"n={n}.txt")
    if os.path.exists(filepath):
        modes = parse_mode_file(filepath)
        all_data[n] = modes
        print(f"n={n}: {len(modes)} 模态")
        for m in modes:
            print(f"  f={m['freq']:.2f} Hz, σ={m['sigma']:.3f}, ζ={m['damping']:.2f}%")

# ==========================================
# 2. 按频率范围归类模态
# ==========================================
mode_bands = {
    '6~12 Hz':    (6, 12),
    '12~18 Hz':   (12, 18),
    '48~62 Hz':   (48, 62),
}
# 匹配参考图的柔和色彩
band_colors = {'6~12 Hz': '#85C1E9', '12~18 Hz': '#F5B041', '48~62 Hz': '#82E0AA'}
# 所有频段均使用圆形散点
band_markers = {'6~12 Hz': 'o', '12~18 Hz': 'o', '48~62 Hz': 'o'}

tracked_modes = {band: {} for band in mode_bands}
for band, (f_low, f_high) in mode_bands.items():
    for n in n_values:
        if n not in all_data:
            tracked_modes[band][n] = None
            continue
        candidates = [m for m in all_data[n] if f_low <= m['freq'] <= f_high]
        if candidates:
            candidates.sort(key=lambda x: abs(x['damping']))
            tracked_modes[band][n] = candidates[0]
        else:
            tracked_modes[band][n] = None

def get_sequence(band, key):
    pairs = [(n, tracked_modes[band][n][key])
             for n in n_values if tracked_modes[band][n] is not None]
    return zip(*pairs) if pairs else ([], [])

# ==========================================
# 3. Fig1: 频率 & 阻尼 vs n (双面板)
# ==========================================
fig1, (ax_f, ax_d) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig1.subplots_adjust(hspace=0.08, right=0.88, top=0.92, bottom=0.10, left=0.12)

for band in mode_bands:
    ns_f, freqs = get_sequence(band, 'freq')
    ns_d, damps = get_sequence(band, 'damping')
    if len(ns_f) == 0:
        continue
    ns_f = list(ns_f); freqs = list(freqs)
    ns_d = list(ns_d); damps = list(damps)

    # 连线
    ax_f.plot(ns_f, freqs, '-', color=band_colors[band], alpha=0.7, linewidth=1.5, zorder=2)
    ax_d.plot(ns_d, damps, '-', color=band_colors[band], alpha=0.7, linewidth=1.5, zorder=2)

    # 散点
    for ni, fi in zip(ns_f, freqs):
        idx = n_values.index(ni)
        m = tracked_modes[band][ni]
        is_unstable = m is not None and m['damping'] < 0
        
        ax_f.scatter(ni, fi, c=[cmap_n(n_norm[idx])],
                     marker='o', s=80, edgecolors='k', linewidths=1.0, zorder=5)
        if is_unstable:
            ax_f.scatter(ni, fi, marker='X', facecolors='yellow', edgecolors='red',
                         s=150, linewidths=1.5, zorder=6)
            
    for ni, di in zip(ns_d, damps):
        idx = n_values.index(ni)
        m = tracked_modes[band][ni]
        is_unstable = m is not None and m['damping'] < 0
        
        ax_d.scatter(ni, di, c=[cmap_n(n_norm[idx])],
                     marker='o', s=80, edgecolors='k', linewidths=1.0, zorder=5)
        if is_unstable:
            ax_d.scatter(ni, di, marker='X', facecolors='yellow', edgecolors='red',
                         s=150, linewidths=1.5, zorder=6)

# ---- 频率子图 (上) ----
ax_f.set_ylabel('Frequency (Hz)', fontsize=12)
ax_f.set_title('Sub-synchronous Mode Frequency vs Number of Wind Turbines', fontsize=14)
ax_f.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax_f.set_xlim(950, 2050)
ax_f.set_ylim(0, 70) 

# 【关键修改点 1】: loc='center right' 放右侧中间，避开所有数据点
legend_band = [Line2D([0], [0], color=band_colors[b], lw=2, label=b) for b in mode_bands]
ax_f.legend(handles=legend_band, loc='center right', title="Mode cluster", fontsize=9, framealpha=0.95)

# ---- 阻尼子图 (下) ----
ax_d.axhline(y=0, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
ax_d.set_xlabel('Number of Wind Turbines n', fontsize=12)
ax_d.set_ylabel('Damping Ratio zeta (%)', fontsize=12)
ax_d.set_title('Sub-synchronous Mode Damping vs Number of Wind Turbines', fontsize=14)
ax_d.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax_d.set_xlim(950, 2050)
ax_d.set_ylim(-15, 100) 

# 【关键修改点 2】: loc='upper right' 放右上角，那里是完全空白的
legend_stab = [Line2D([0], [0], color=band_colors[b], lw=1.5, alpha=0.7, label=b) for b in mode_bands]
legend_stab.append(Line2D([0], [0], color='red', lw=1.5, linestyle='--', alpha=0.5, label='Stability boundary (zeta = 0)'))
ax_d.legend(handles=legend_stab, loc='upper right', fontsize=9, framealpha=0.95)

# 隐藏上子图 x 轴标签
ax_f.tick_params(axis='x', labelbottom=False)

# ---- 颜色条 (右侧) ----
cbar_ax = fig1.add_axes([0.90, 0.15, 0.02, 0.70])
sm = plt.cm.ScalarMappable(cmap=cmap_n, norm=plt.Normalize(vmin=1000, vmax=2000))
sm.set_array([])
cbar = fig1.colorbar(sm, cax=cbar_ax)
cbar.set_label('n (turbines)', fontsize=12)

fig1.savefig(os.path.join(output_dir, 'fig1_freq_damping_vs_n.png'), dpi=300, bbox_inches='tight')

# ==========================================
# 4. Fig2: s-平面零点迁移轨迹
# ==========================================
fig2, ax_s = plt.subplots(figsize=(11, 9))
fig2.subplots_adjust(right=0.90, top=0.92, bottom=0.10, left=0.12)

# ---- 稳定/不稳定区域背景 ----
# 根据参考图调整绿红阴影带范围及不透明度
ax_s.axvspan(-5, 0, alpha=0.05, color='green', zorder=0)
ax_s.axvspan(0, 15, alpha=0.03, color='red', zorder=0)
ax_s.axvline(x=0, color='gray', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)

# 文字标注
ax_s.text(-1, 370, 'STABLE', fontsize=11, color='green', alpha=0.4, fontweight='bold', ha='right')
ax_s.text(1, 370, 'UNSTABLE', fontsize=11, color='red', alpha=0.4, fontweight='bold', ha='left')

for band in mode_bands:
    points = []
    for n in n_values:
        m = tracked_modes[band][n]
        if m is not None:
            points.append((m['sigma'], m['omega'], n, m))
    
    # 绘制带弧度的迁移箭头 (arc3, rad=0.1)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            s1, o1, _, _ = points[i]
            s2, o2, _, _ = points[i + 1]
            ax_s.annotate('', xy=(s2, o2), xytext=(s1, o1),
                          arrowprops=dict(arrowstyle='->', color=band_colors[band],
                                          lw=1.0, alpha=0.7, 
                                          connectionstyle='arc3,rad=0.1', zorder=2))

    for sigma, omega, n, m in points:
        idx = n_values.index(n)
        is_unstable = m['damping'] < 0
        
        ax_s.scatter(sigma, omega, c=[cmap_n(n_norm[idx])],
                     marker='o', s=100, edgecolors='k', linewidths=1.0, zorder=5)
        if is_unstable:
            ax_s.scatter(sigma, omega, marker='X', facecolors='yellow', edgecolors='red',
                         s=180, linewidths=1.5, zorder=6)

    # 仅标注 n=2000 的文字，去掉 "Stable" 前缀以匹配参考图
    m_end = tracked_modes[band][2000]
    if m_end is not None:
        label = f"{m_end['freq']:.1f} Hz"
        # 为不同频率设置合理的偏移以避免遮挡
        offset = (15, 10) if m_end['omega'] > 100 else (15, 5)
        ax_s.annotate(label, (m_end['sigma'], m_end['omega']),
                      textcoords="offset points", xytext=offset,
                      fontsize=9, fontweight='bold', color='black')

ax_s.set_xlabel('Real Part sigma (1/s)', fontsize=13)
ax_s.set_ylabel('Imaginary Part omega (rad/s)', fontsize=13)
ax_s.set_title('Sub-synchronous Modes Migration in s-Plane\n(8Hz / 14Hz / 55Hz, n = 1000 ~ 2000)', fontsize=14)
ax_s.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 根据参考图调整轴限制
ax_s.set_xlim(-125, 15)
ax_s.set_ylim(-15, 390)

# ---- 图例 (双列自定义图例) ----
legend_elements = []
for i, n in enumerate(n_values):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_n(n_norm[i]), 
                                  markersize=8, markeredgecolor='k', label=f'n={n}'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                              markersize=9, markeredgecolor='k', label='Stable'))
legend_elements.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='yellow', markeredgecolor='red', 
                              markersize=10, markeredgewidth=1.5, label='Unstable'))

ax_s.legend(handles=legend_elements, loc='lower right', ncol=2, fontsize=8, framealpha=1.0)

fig2.savefig(os.path.join(output_dir, 'fig2_splane_migration.png'), dpi=300, bbox_inches='tight')

# ==========================================
# 5. 模态汇总表 (保持不变)
# ==========================================
plt.show()