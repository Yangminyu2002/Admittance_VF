# Admittance_VF

基于 PSCAD/AIM 频率扫描数据的并网系统导纳稳定性分析与 Vector Fitting 模态识别脚本集。

本项目主要用于读取 dq 坐标系下的 MIMO 导纳数据，计算系统导纳矩阵行列式 $\det(Y_{LIM})$，再通过 Vector Fitting 提取零点、模态频率和阻尼比，并绘制不同运行参数（风机数量、风速）下的次同步模态迁移图。

## 功能概览

- 读取 AIM 导出的 `T1/T2_admittance_dq_MIMO1.txt` 频扫数据。
- 将幅值（dB）、相位角（deg）转换为复数导纳。
- 计算两端导纳叠加后的 $Y_{LIM} = Y_s + Y_g$。
- 计算 $\det(Y_{LIM})$ 并输出实部、虚部、幅值和相位。
- 绘制 $\det(Y_{LIM})$ Bode 图及 $Y_{LIM}$ MIMO 各分量 Bode 图。
- 使用 Vector Fitting（支持普通模式和共轭约束模式）对 $\det(Y_{LIM})$ 进行有理函数拟合。
- 从拟合模型中提取零点，计算模态频率 $f$ 和阻尼比 $\zeta$。
- 绘制拟合对比图、频率/阻尼变化图和 s 平面零点迁移轨迹。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `admittance_Ylim_fitting.py` | **一体化脚本（推荐）**：读取两端 dq 导纳数据，计算 $Y_{LIM}$ 和 $\det(Y_{LIM})$，绘制 Bode 图，执行 Vector Fitting 并输出模态结果。将计算与拟合步骤合并在一个脚本中。 |
| `admittance_Ylim.py` | **独立计算脚本**：读取两端 dq 导纳数据，计算 $Y_{LIM}$ 和 $\det(Y_{LIM})$，绘制 Bode 图并保存 `det_Ylim_results.txt`。 |
| `Ylim_fitting.py` | **独立拟合脚本**：读取 `det_Ylim_results.txt`，执行 Vector Fitting，输出拟合传递函数、零点模态表和拟合对比图。 |
| `VF_lib.py` | Vector Fitting 核心函数库，包含 `vector_fitting_zeros`（通用模式）和 `vector_fitting_zeros_conjugate`（共轭约束模式）两个版本。 |
| `zero_trajectory_n.py` | 根据不同风机数量 $n$ 下的模态识别结果，绘制频率/阻尼变化图和 s 平面零点迁移轨迹图。 |
| `zero_trajectory_vw.py` | 根据不同风速 $v_w$ 下的模态识别结果，绘制频率/阻尼变化图和 s 平面零点迁移轨迹图。 |

## 环境依赖

Python 3.9+，依赖 `numpy`、`pandas`、`matplotlib`。

```powershell
pip install numpy pandas matplotlib
```

Conda 环境（推荐）：

```powershell
conda create -n plot_env python=3.11 numpy pandas matplotlib
conda activate plot_env
```

## 使用流程

### 方式一：一体化脚本（推荐）

直接使用 `admittance_Ylim_fitting.py`，修改脚本顶部的配置参数后运行：

```powershell
python admittance_Ylim_fitting.py
```

该脚本中的可配置参数：

| 参数 | 说明 |
| --- | --- |
| `TERMINAL2_ADMITTANCE_FILE` | Terminal2（源侧）dq 导纳文件路径 |
| `TERMINAL1_ADMITTANCE_FILE` | Terminal1（电网侧）dq 导纳文件路径 |
| `OUTPUT_DIR` | 所有输出文件的保存目录 |
| `N_ORDER` | Vector Fitting 拟合阶数（默认 6） |
| `FIT_NROWS` | 拟合使用的频率点数（默认 120） |
| `USE_CONJUGATE_FITTING` | 是否使用共轭约束拟合（默认 `False`） |
| `SHOW_PLOTS` | 是否显示绘图窗口（默认 `True`） |

### 方式二：分步执行

1. **准备数据**：准备 AIM/PSCAD 导出的 dq 导纳频扫文件。输入文件需包含频率列 `fp`，以及以下幅值/相位列：
   ```
   Ydd_mag Ydd_pha  Ydq_mag Ydq_pha
   Yqd_mag Yqd_pha  Yqq_mag Yqq_pha
   ```

2. **计算 $\det(Y_{LIM})$**：修改 `admittance_Ylim.py` 中的输入路径，运行：
   ```powershell
   python admittance_Ylim.py
   ```
   生成 `det_Ylim_results.txt`，字段包括：
   ```
   Frequency_Hz Real_Part Imag_Part Magnitude Phase_deg
   ```

3. **Vector Fitting 拟合**：修改 `Ylim_fitting.py` 中的 `det_Ylim_results.txt` 路径，运行：
   ```powershell
   python Ylim_fitting.py
   ```
   输出：
   - `VF_fit_TransferFunction_and_Modes.txt` — 传递函数解析表达式和模态表
   - `VF_fit_plot.png` — 拟合对比图
   - 控制台打印的零点模态表

### 绘制模态迁移图

将不同工况下生成的 `VF_fit_TransferFunction_and_Modes.txt` 整理为约定的文件名格式，然后运行轨迹绘制脚本。

**风机数量 $n$ 变化**（`zero_trajectory_n.py`）：
- 将各工况文件重命名为 `n=1000.txt`、`n=1250.txt`、`n=1500.txt` 等
- 修改脚本中的 `data_dir` 为文件所在目录
- 运行：`python zero_trajectory_n.py`

**风速 $v_w$ 变化**（`zero_trajectory_vw.py`）：
- 将各工况文件重命名为 `vw=8.txt`、`vw=8.5.txt`、`vw=9.txt` 等
- 修改脚本中的 `data_dir` 为文件所在目录
- 运行：`python zero_trajectory_vw.py`

输出图片包括：
- `fig1_freq_damping_vs_n.png` / `fig1_freq_damping_vs_vw.png` — 频率和阻尼随参数变化的双面板图
- `fig2_s_plane_trajectory_n.png` / `fig2_s_plane_trajectory_vw.png` — s 平面零点迁移轨迹图

## Vector Fitting 算法说明

`VF_lib.py` 提供两种拟合模式：

| 函数 | 说明 |
| --- | --- |
| `vector_fitting_zeros(s, y, n_order)` | 标准 Vector Fitting，采用极点重定位迭代，通过极点-留数模型提取零点。 |
| `vector_fitting_zeros_conjugate(s, y, n_order)` | 共轭约束版本，强制极点/留数为严格共轭对，常数项 $d$ 为实数，适合物理系统的有理拟合。 |

两种模式均返回：极点 `poles`、零点 `zeros`、拟合曲线 `y_fit`、留数 `residues` 和常数项 `d`。

零点提取基于状态空间法：有理函数 $f(s) = d + \sum \frac{r_i}{s-p_i}$ 的零点为矩阵 $\text{diag}(p_i) - \frac{1}{d} \mathbf{1} \cdot \mathbf{r}^T$ 的特征值。

## 主要输出

- `det_Ylim_results.txt`：`det(Y_LIM)` 的频域复数结果。
- `VF_fit_TransferFunction_and_Modes.txt`：Vector Fitting 拟合表达式和模态表。
- `VF_fit_plot.png`：原始数据与拟合结果对比图。
- `fig1_freq_damping_vs_n.png`：模态频率/阻尼随风机数量变化图。
- `fig2_splane_migration.png`：风机数量变化下的 s 平面零点迁移图。
- `fig1_freq_damping_vs_vw.png`：模态频率/阻尼随风速变化图。
- `fig2_splane_migration_vw.png`：风速变化下的 s 平面零点迁移图。

## 方法说明

对于 2x2 dq 导纳矩阵：

```text
Y_LIM = [[Ydd, Ydq],
         [Yqd, Yqq]]
```

脚本计算：

```text
det(Y_LIM) = Ydd * Yqq - Ydq * Yqd
```

随后将频域响应写为 `s = j * 2*pi*f`，使用有理函数形式拟合：

```text
f(s) = d + sum(r_i / (s - p_i))
```

其中 `p_i` 为拟合极点，`r_i` 为留数，`d` 为常数项。脚本进一步通过状态空间等效形式提取拟合函数零点，并根据零点计算：

```text
Freq_Hz = imag(zero) / (2*pi)
Damping_Ratio_% = -real(zero) / abs(zero) * 100
```

阻尼比小于 0 的模态在迁移图中标记为不稳定模态。

## 注意事项

- 当前脚本中仍包含若干硬编码的本机路径，运行前需要按自己的数据目录修改。
- 项目中的注释存在部分编码显示异常，但不影响核心计算逻辑。
- `zero_trajectory_n.py` 和 `zero_trajectory_vw.py` 依赖前一步拟合输出的模态表文本格式。
- 若运行环境没有图形界面，可将 `plt.show()` 注释掉，仅保留 `savefig()` 输出图片。

## Git

本项目已初始化为 Git 仓库，并推送到：

```text
https://github.com/Yangminyu2002/Admittance_VF.git
```
