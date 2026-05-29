import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import os

# ==========================================
# 0. 全局设置
# ==========================================
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

data_dir = r"d:\Python\damping_modes_vw"
output_dir = data_dir

vw_values = [8, 8.5, 9, 9.5, 10]
vw_norm = (np.array(vw_values) - min(vw_values)) / (max(vw_values) - min(vw_values))
cmap_vw = plt.cm.plasma

# ==========================================
# 1. 解析各 vw 值的模态数据
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
for vw in vw_values:
    filepath = os.path.join(data_dir, f"vw={vw}.txt")
    if os.path.exists(filepath):
        modes = parse_mode_file(filepath)
        all_data[vw] = modes
        print(f"vw={vw}: {len(modes)} 模态")
        for m in modes:
            print(f"  f={m['freq']:.2f} Hz, σ={m['sigma']:.3f}, ζ={m['damping']:.2f}%")

# ==========================================
# 2. 按频率范围归类模态
# ==========================================
mode_bands = {
    '13~15 Hz':  (12, 17),
    '55~58 Hz':  (54, 60),
}
band_colors = {'13~15 Hz': '#85C1E9', '55~58 Hz': '#82E0AA'}

tracked_modes = {band: {} for band in mode_bands}
for band, (f_low, f_high) in mode_bands.items():
    for vw in vw_values:
        if vw not in all_data:
            tracked_modes[band][vw] = None
            continue
        candidates = [m for m in all_data[vw] if f_low <= m['freq'] <= f_high]
        if candidates:
            candidates.sort(key=lambda x: abs(x['damping']))
            tracked_modes[band][vw] = candidates[0]
        else:
            tracked_modes[band][vw] = None

def get_sequence(band, key):
    pairs = [(vw, tracked_modes[band][vw][key])
             for vw in vw_values if tracked_modes[band][vw] is not None]
    return zip(*pairs) if pairs else ([], [])

# ==========================================
# 3. Fig1: 频率 & 阻尼 vs vw (双面板)
# ==========================================
fig1, (ax_f, ax_d) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig1.subplots_adjust(hspace=0.08, right=0.88, top=0.92, bottom=0.10, left=0.12)

for band in mode_bands:
    vws_f, freqs = get_sequence(band, 'freq')
    vws_d, damps = get_sequence(band, 'damping')
    if len(vws_f) == 0:
        continue
    vws_f = list(vws_f); freqs = list(freqs)
    vws_d = list(vws_d); damps = list(damps)

    # 连线
    ax_f.plot(vws_f, freqs, '-', color=band_colors[band], alpha=0.7, linewidth=1.5, zorder=2)
    ax_d.plot(vws_d, damps, '-', color=band_colors[band], alpha=0.7, linewidth=1.5, zorder=2)

    # 散点
    for vi, fi in zip(vws_f, freqs):
        idx = vw_values.index(vi)
        m = tracked_modes[band][vi]
        is_unstable = m is not None and m['damping'] < 0

        ax_f.scatter(vi, fi, c=[cmap_vw(vw_norm[idx])],
                     marker='o', s=80, edgecolors='k', linewidths=1.0, zorder=5)
        if is_unstable:
            ax_f.scatter(vi, fi, marker='X', facecolors='yellow', edgecolors='red',
                         s=150, linewidths=1.5, zorder=6)

    for vi, di in zip(vws_d, damps):
        idx = vw_values.index(vi)
        m = tracked_modes[band][vi]
        is_unstable = m is not None and m['damping'] < 0

        ax_d.scatter(vi, di, c=[cmap_vw(vw_norm[idx])],
                     marker='o', s=80, edgecolors='k', linewidths=1.0, zorder=5)
        if is_unstable:
            ax_d.scatter(vi, di, marker='X', facecolors='yellow', edgecolors='red',
                         s=150, linewidths=1.5, zorder=6)

# ---- 频率子图 (上) ----
ax_f.set_ylabel('Frequency (Hz)', fontsize=12)
ax_f.set_title('Sub-synchronous Mode Frequency vs Wind Speed', fontsize=14)
ax_f.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax_f.set_xlim(7.8, 10.2)
ax_f.set_ylim(0, 70)

legend_band = [Line2D([0], [0], color=band_colors[b], lw=2, label=b) for b in mode_bands]
ax_f.legend(handles=legend_band, loc='center right', title="Mode cluster", fontsize=9, framealpha=0.95)

# ---- 阻尼子图 (下) ----
ax_d.axhline(y=0, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
ax_d.set_xlabel('Wind Speed vw (m/s)', fontsize=12)
ax_d.set_ylabel('Damping Ratio zeta (%)', fontsize=12)
ax_d.set_title('Sub-synchronous Mode Damping vs Wind Speed', fontsize=14)
ax_d.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax_d.set_xlim(7.8, 10.2)
ax_d.set_ylim(-10, 40)

legend_stab = [Line2D([0], [0], color=band_colors[b], lw=1.5, alpha=0.7, label=b) for b in mode_bands]
legend_stab.append(Line2D([0], [0], color='red', lw=1.5, linestyle='--', alpha=0.5, label='Stability boundary (zeta = 0)'))
ax_d.legend(handles=legend_stab, loc='upper right', fontsize=9, framealpha=0.95)

ax_f.tick_params(axis='x', labelbottom=False)

# ---- 颜色条 (右侧) ----
cbar_ax = fig1.add_axes([0.90, 0.15, 0.02, 0.70])
sm = plt.cm.ScalarMappable(cmap=cmap_vw, norm=plt.Normalize(vmin=8, vmax=10))
sm.set_array([])
cbar = fig1.colorbar(sm, cax=cbar_ax)
cbar.set_label('vw (m/s)', fontsize=12)

fig1.savefig(os.path.join(output_dir, 'fig1_freq_damping_vs_vw.png'), dpi=300, bbox_inches='tight')

# ==========================================
# 4. Fig2: s-平面零点迁移轨迹
# ==========================================
fig2, ax_s = plt.subplots(figsize=(11, 9))
fig2.subplots_adjust(right=0.90, top=0.92, bottom=0.10, left=0.12)

# ---- 稳定/不稳定区域背景 ----
ax_s.axvspan(-120, 0, alpha=0.05, color='green', zorder=0)
ax_s.axvspan(0, 10, alpha=0.03, color='red', zorder=0)
ax_s.axvline(x=0, color='gray', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)

# 文字标注
ax_s.text(-3, 375, 'STABLE', fontsize=11, color='green', alpha=0.4, fontweight='bold', ha='right')
ax_s.text(3, 375, 'UNSTABLE', fontsize=11, color='red', alpha=0.4, fontweight='bold', ha='left')

for band in mode_bands:
    points = []
    for vw in vw_values:
        m = tracked_modes[band][vw]
        if m is not None:
            points.append((m['sigma'], m['omega'], vw, m))

    # 绘制带弧度的迁移箭头
    if len(points) >= 2:
        for i in range(len(points) - 1):
            s1, o1, _, _ = points[i]
            s2, o2, _, _ = points[i + 1]
            ax_s.annotate('', xy=(s2, o2), xytext=(s1, o1),
                          arrowprops=dict(arrowstyle='->', color=band_colors[band],
                                          lw=1.0, alpha=0.7,
                                          connectionstyle='arc3,rad=0.1', zorder=2))

    for sigma, omega, vw, m in points:
        idx = vw_values.index(vw)
        is_unstable = m['damping'] < 0

        ax_s.scatter(sigma, omega, c=[cmap_vw(vw_norm[idx])],
                     marker='o', s=100, edgecolors='k', linewidths=1.0, zorder=5)
        if is_unstable:
            ax_s.scatter(sigma, omega, marker='X', facecolors='yellow', edgecolors='red',
                         s=180, linewidths=1.5, zorder=6)

    # 标注 vw=10 (终点) 的频率
    m_end = tracked_modes[band][10]
    if m_end is not None:
        label = f"{m_end['freq']:.1f} Hz"
        offset = (15, 10) if m_end['omega'] > 200 else (15, 5)
        ax_s.annotate(label, (m_end['sigma'], m_end['omega']),
                      textcoords="offset points", xytext=offset,
                      fontsize=9, fontweight='bold', color='black')

ax_s.set_xlabel('Real Part sigma (1/s)', fontsize=13)
ax_s.set_ylabel('Imaginary Part omega (rad/s)', fontsize=13)
ax_s.set_title('Sub-synchronous Modes Migration in s-Plane\n(14Hz / 57Hz, vw = 8 ~ 10 m/s)', fontsize=14)
ax_s.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

ax_s.set_xlim(-125, 15)
ax_s.set_ylim(-15, 390)

# ---- 图例 (双列) ----
legend_elements = []
for i, vw in enumerate(vw_values):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_vw(vw_norm[i]),
                                  markersize=8, markeredgecolor='k', label=f'vw={vw}'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              markersize=9, markeredgecolor='k', label='Stable'))
legend_elements.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='yellow', markeredgecolor='red',
                              markersize=10, markeredgewidth=1.5, label='Unstable'))

ax_s.legend(handles=legend_elements, loc='lower right', ncol=2, fontsize=8, framealpha=1.0)

fig2.savefig(os.path.join(output_dir, 'fig2_splane_migration_vw.png'), dpi=300, bbox_inches='tight')

plt.show()
