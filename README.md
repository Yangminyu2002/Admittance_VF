# Admittance_VF

基于 PSCAD/AIM 频率扫描数据的并网系统导纳稳定性分析与 Vector Fitting 模态识别脚本集。

本项目主要用于读取 dq 坐标系下的 MIMO 导纳数据，计算系统导纳矩阵行列式 `det(Y_LIM)`，再通过 Vector Fitting 提取零点、模态频率和阻尼比，并绘制不同运行参数下的次同步模态迁移图。

## 功能概览

- 读取 AIM 导出的 `T1/T2_admittance_dq_MIMO1.txt` 频扫数据。
- 将幅值 dB、相位角转换为复数导纳。
- 计算两端导纳叠加后的 `Y_LIM = Y_s + Y_g`。
- 计算 `det(Y_LIM)` 并输出实部、虚部、幅值和相位。
- 使用 Vector Fitting 对 `det(Y_LIM)` 进行有理函数拟合。
- 从拟合模型中提取零点，并计算模态频率和阻尼比。
- 绘制拟合对比图、频率/阻尼变化图和 s 平面零点迁移轨迹。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `admittance_Ylim.py` | 读取两端 dq 导纳数据，计算 `Y_LIM` 和 `det(Y_LIM)`，绘制 Bode 图并保存结果。 |
| `Ylim_fitting.py` | 读取 `det_Ylim_results.txt`，执行 Vector Fitting，输出拟合传递函数、零点模态表和拟合对比图。 |
| `VF_lib.py` | Vector Fitting 核心函数库，包含普通零点提取和共轭约束拟合版本。 |
| `zero_trajectory_n.py` | 根据不同风机数量 `n` 下的模态识别结果，绘制频率/阻尼变化和 s 平面迁移图。 |
| `zero_trajectory_vw.py` | 根据不同风速 `vw` 下的模态识别结果，绘制频率/阻尼变化和 s 平面迁移图。 |

## 环境依赖

建议使用 Python 3.9 或更高版本。

```powershell
pip install numpy pandas matplotlib
```

如果使用 Conda，也可以创建独立环境：

```powershell
conda create -n admittance-vf python=3.11 numpy pandas matplotlib
conda activate admittance-vf
```

## 典型使用流程

1. 准备 AIM/PSCAD 导出的 dq 导纳频扫文件。

   输入文件需包含频率列 `fp`，以及以下幅值/相位列：

   ```text
   Ydd_mag Ydd_pha
   Ydq_mag Ydq_pha
   Yqd_mag Yqd_pha
   Yqq_mag Yqq_pha
   ```

2. 修改 `admittance_Ylim.py` 中的输入路径。

   需要将脚本中的 `T1_admittance_dq_MIMO1.txt` 和 `T2_admittance_dq_MIMO1.txt` 路径改为本机实际数据路径。

3. 计算 `det(Y_LIM)`。

   ```powershell
   python admittance_Ylim.py
   ```

   该步骤会生成类似 `det_Ylim_results.txt` 的结果文件，字段包括：

   ```text
   Frequency_Hz Real_Part Imag_Part Magnitude Phase_deg
   ```

4. 修改 `Ylim_fitting.py` 中的 `det_Ylim_results.txt` 路径，然后执行拟合。

   ```powershell
   python Ylim_fitting.py
   ```

   输出内容包括：

   - `VF_fit_TransferFunction_and_Modes.txt`
   - `VF_fit_plot.png`
   - 控制台打印的零点模态表

5. 绘制参数变化下的模态迁移。

   将不同工况下生成的 `VF_fit_TransferFunction_and_Modes.txt` 整理为脚本中约定的文件名，例如：

   ```text
   n=1000.txt
   n=1250.txt
   n=1500.txt
   vw=8.txt
   vw=8.5.txt
   ```

   然后修改 `zero_trajectory_n.py` 或 `zero_trajectory_vw.py` 中的 `data_dir`，运行：

   ```powershell
   python zero_trajectory_n.py
   python zero_trajectory_vw.py
   ```

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
