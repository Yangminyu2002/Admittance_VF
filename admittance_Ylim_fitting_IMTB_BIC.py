import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import VF_lib


# IMTB data file — contains both Zdut and Znet in dq frame.
IMTB_DATA_FILE = (
    r"IMTB_data\20260602165858_n=2000_X=0.5"
    r"\IM_Zdq_MIMO_n=2000_X=0.5_posneg.csv"
)

# All output files are written here.
OUTPUT_DIR = r"IMTB_data\20260602165858_n=2000_X=0.5\Ylim_results"

# Vector Fitting settings.
N_ORDER = 6
FIT_NROWS = 120
USE_CONJUGATE_FITTING = False
SHOW_PLOTS = True

# BIC sweep settings.
BIC_SWEEP_MIN_ORDER = 2
BIC_SWEEP_MAX_ORDER = 20
BIC_SWEEP_STEP = 1
BIC_PENALTY_FACTOR = 1.0   # >1 → stricter penalty, smaller order


# =============================================================================
#  Data loading  (IMTB format: Z → Y via 2×2 matrix inversion)
# =============================================================================

def z_to_y(z_11, z_12, z_21, z_22):
    """Convert 2×2 impedance matrix Z to admittance matrix Y = Z⁻¹.

    For each frequency point:
        det(Z) = z_11*z_22 - z_12*z_21
        Y = 1/det(Z) * [[z_22, -z_12], [-z_21, z_11]]
    """
    det = z_11 * z_22 - z_12 * z_21
    return {
        "dd": z_22 / det,
        "dq": -z_12 / det,
        "qd": -z_21 / det,
        "qq": z_11 / det,
    }


def load_imtb_admittance(filename):
    """Read IMTB dq impedance CSV and convert to complex admittance matrices.

    The CSV columns follow the pattern:
        Zdut_dq_{ij}_{re|im}   — device under test impedance
        Znet_dq_{ij}_{re|im}   — network impedance
    where ij ∈ {11, 12, 21, 22} maps to dd, dq, qd, qq.
    """
    df = pd.read_csv(filename)
    # Keep only positive frequencies.
    df = df[df["f"] >= 0].reset_index(drop=True)
    freq = df["f"].values

    comp_map = {"11": "dd", "12": "dq", "21": "qd", "22": "qq"}

    # --- Zdut ---
    z_dut = {}
    for ij, comp in comp_map.items():
        re_col = f"Zdut_dq_{ij}_re"
        im_col = f"Zdut_dq_{ij}_im"
        z_dut[comp] = df[re_col].values + 1j * df[im_col].values

    y_dut = z_to_y(z_dut["dd"], z_dut["dq"], z_dut["qd"], z_dut["qq"])

    # --- Znet ---
    z_net = {}
    for ij, comp in comp_map.items():
        re_col = f"Znet_dq_{ij}_re"
        im_col = f"Znet_dq_{ij}_im"
        z_net[comp] = df[re_col].values + 1j * df[im_col].values

    y_net = z_to_y(z_net["dd"], z_net["dq"], z_net["qd"], z_net["qq"])

    return freq, y_dut, y_net


def calculate_y_lim(imtb_file):
    """Calculate Y_LIM = Y_dut + Y_net and det(Y_LIM)."""
    freq, y_dut, y_net = load_imtb_admittance(imtb_file)

    y_lim = {}
    for component in ["dd", "dq", "qd", "qq"]:
        y_lim[component] = y_dut[component] + y_net[component]

    det_y_lim = y_lim["dd"] * y_lim["qq"] - y_lim["dq"] * y_lim["qd"]
    return freq, y_lim, det_y_lim


# =============================================================================
#  Save / Plot  (same as original)
# =============================================================================

def save_det_y_lim(freq, det_y_lim, output_dir):
    """Save determinant data in the format consumed by the fitting step."""
    os.makedirs(output_dir, exist_ok=True)

    magnitude = np.abs(det_y_lim)
    phase_deg = np.rad2deg(np.angle(det_y_lim))
    output_df = pd.DataFrame(
        {
            "Frequency_Hz": freq,
            "Real_Part": np.real(det_y_lim),
            "Imag_Part": np.imag(det_y_lim),
            "Magnitude": magnitude,
            "Phase_deg": phase_deg,
        }
    )

    output_path = os.path.join(output_dir, "det_Ylim_results.txt")
    output_df.to_csv(output_path, sep=" ", index=False, float_format="%.6e")
    print(f"[OK] det(Y_LIM) saved to: {output_path}")
    return output_path


def plot_det_bode(freq, det_y_lim, output_dir):
    mag_db = 20 * np.log10(np.abs(det_y_lim))
    phase_deg = np.rad2deg(np.angle(det_y_lim))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(r"Bode Plot of $\det(Y_{LIM})$", fontsize=16)

    ax1.plot(freq, mag_db, color="g", linewidth=1.5, label=r"$|\det(Y_{LIM})|$")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend(loc="upper right")
    ax1.set_xlim([min(freq), max(freq)])

    ax2.plot(freq, phase_deg, color="orange", linewidth=1.5, label=r"Phase of $\det(Y_{LIM})$")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend(loc="upper right")
    ax2.set_xlim([min(freq), max(freq)])

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    png_path = os.path.join(output_dir, "det_Ylim_bode.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[OK] det(Y_LIM) Bode plot saved to: {png_path}")


def plot_bode_mimo_linear(freq, y_lim, output_dir):
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    layout_map = [
        ("dd", "Ydd(s)"),
        ("dq", "Ydq(s)"),
        ("dd", "Ydd(s)"),
        ("dq", "Ydq(s)"),
        ("qd", "Yqd(s)"),
        ("qq", "Yqq(s)"),
        ("qd", "Yqd(s)"),
        ("qq", "Yqq(s)"),
    ]

    for i, (key, title) in enumerate(layout_map):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        if row % 2 == 0:
            mag_db = 20 * np.log10(np.abs(y_lim[key]))
            ax.plot(freq, mag_db, color="tab:blue", linewidth=1.5)
            ax.set_ylabel("Mag (dB)")
            ax.set_title(title, fontweight="bold")
        else:
            phase_deg = np.rad2deg(np.angle(y_lim[key]))
            ax.plot(freq, phase_deg, color="tab:red", linewidth=1.5)
            ax.set_ylabel("Phase (deg)")

        ax.grid(True, linestyle="--", alpha=0.7)
        if row == 3:
            ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout()

    png_path = os.path.join(output_dir, "Ylim_mimo_bode.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Y_LIM MIMO Bode plot saved to: {png_path}")


# =============================================================================
#  BIC  /  Vector Fitting
# =============================================================================

def compute_bic(y_original, y_fit, n_params):
    """Compute the Bayesian Information Criterion (BIC).

    Uses the form:
        BIC = m * ln(ε_mean) + ln(m) * n_params

    where:
        m       = number of data points (complex)
        ε_mean  = mean squared error  (1/m) * Σ |y_orig - y_fit|²
        n_params = number of real-valued free parameters

    BIC penalises model complexity more heavily than AIC, favouring
    simpler models.
    """
    m = len(y_original)  # number of complex data points
    squared_errors = np.abs(y_original - y_fit) ** 2
    epsilon_mean = np.mean(squared_errors)
    if epsilon_mean <= 0:
        return np.inf
    bic = m * np.log(epsilon_mean) + BIC_PENALTY_FACTOR * np.log(m) * n_params
    return bic


def n_params_for_order(n_order):
    """Number of real-valued parameters for a vector fit of given order.

    The model  f(s) = d + Σ r_k / (s - p_k)  has:
        - n_order complex poles    → 2 * n_order real params
        - n_order complex residues → 2 * n_order real params
        - 1 complex d-term         → 2 real params
        Total: 4 * n_order + 2
    """
    return 4 * n_order + 2


def build_modes_table(zeros, tol=1e-9):
    """Pair conjugate zeros into 'σ ± jω' rows; real zeros keep just σ.

    Returns a DataFrame with columns: Mode, Freq_Hz, Damping_%
    """
    real_zeros = []
    imag_pos = []  # (sigma, +omega)

    for z in zeros:
        s, w = z.real, z.imag
        if abs(w) < tol:
            real_zeros.append(s)
        elif w > 0:
            imag_pos.append((s, w))
        # w < 0 is the conjugate partner — skip, already paired via imag_pos

    rows = []
    for s, w in sorted(imag_pos, key=lambda x: x[1]):
        freq = w / (2 * np.pi)
        damp = -s / np.sqrt(s**2 + w**2) * 100
        rows.append((f"{s:.6e} ± j{w:.6e}", freq, damp))

    for s in sorted(real_zeros):
        rows.append((f"{s:.6e}", np.nan, 100.0))

    if not rows:
        return pd.DataFrame(columns=["Mode (σ ± jω)", "Freq_Hz", "Damping_%"])

    return pd.DataFrame(rows, columns=["Mode (σ ± jω)", "Freq_Hz", "Damping_%"])


def build_analytical_expression(poles, residues, d_const, var_name="s"):
    expr_parts = [
        f"{d_const.real:.4e} + {d_const.imag:.4e}j"
        if d_const.imag != 0
        else f"{d_const.real:.4e}"
    ]
    for pole, residue in zip(poles, residues):
        residue_str = f"({residue.real:.4e} + {residue.imag:.4e}j)"
        pole_str = f"({pole.real:.4e} + {pole.imag:.4e}j)"
        expr_parts.append(f"{residue_str} / ({var_name} - {pole_str})")
    return " + ".join(expr_parts)


def plot_s_plane(poles, zeros, output_dir, n_order):
    """Plot zeros (○) on the s-plane: full view + zoomed view near jω axis."""
    # ---- Figure 1: full view ----
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.plot(zeros.real, zeros.imag, "ro", markersize=8, markerfacecolor="none",
             markeredgewidth=1.5, label="Zeros")
    ax1.set_xlabel("Real (σ)")
    ax1.set_ylabel("Imag (jω)")
    ax1.set_title(f"Zero Map (s-plane, full view)  —  order N={n_order}")
    ax1.legend(loc="upper right")
    ax1.grid(True, ls="--", alpha=0.5)
    re_range = np.ptp(zeros.real) or 1.0
    im_range = np.ptp(zeros.imag) or 1.0
    m = 0.10
    ax1.set_xlim(zeros.real.min() - m * re_range, zeros.real.max() + m * re_range)
    ax1.set_ylim(zeros.imag.min() - m * im_range, zeros.imag.max() + m * im_range)
    plt.tight_layout()
    png1 = os.path.join(output_dir, "s_plane_zeros_full.png")
    fig1.savefig(png1, dpi=300, bbox_inches="tight")
    print(f"[OK] s-plane zero map (full) saved to: {png1}")

    # ---- Figure 2: zoomed near jω axis (IQR-based) ----
    def _iqr_range(values, factor=2.0):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1 or 1.0
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        pad = (hi - lo) * 0.10 or 1.0
        return lo - pad, hi + pad

    x1, x2 = _iqr_range(zeros.real)
    y1, y2 = _iqr_range(zeros.imag)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.plot(zeros.real, zeros.imag, "ro", markersize=8, markerfacecolor="none",
             markeredgewidth=1.5, label="Zeros")
    ax2.set_xlabel("Real (σ)")
    ax2.set_ylabel("Imag (jω)")
    ax2.set_title(f"Zero Map (s-plane, zoomed)  —  order N={n_order}")
    ax2.legend(loc="upper right")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    plt.tight_layout()
    png2 = os.path.join(output_dir, "s_plane_zeros_zoom.png")
    fig2.savefig(png2, dpi=300, bbox_inches="tight")
    print(f"[OK] s-plane zero map (zoom)  saved to: {png2}")


def fit_det_y_lim(det_results_file, output_dir, n_order=None, silent=False):
    """Vector-fit det(Y_LIM) with a single order; report BIC.

    When silent=True (used during BIC sweep), individual fit plots and
    verbose output are suppressed.
    """
    if n_order is None:
        n_order = N_ORDER

    if USE_CONJUGATE_FITTING and n_order % 2 != 0:
        raise ValueError(
            f"Conjugate fitting requires an even order, got {n_order}."
        )

    df = pd.read_csv(det_results_file, sep=r"\s+", nrows=FIT_NROWS)
    freq = df["Frequency_Hz"].values
    s = 1j * 2 * np.pi * freq
    det_y = df["Real_Part"].values + 1j * df["Imag_Part"].values

    if USE_CONJUGATE_FITTING:
        poles, zeros, det_y_fit, residues, d_const = VF_lib.vector_fitting_zeros_conjugate(
            s, det_y, n_order
        )
    else:
        poles, zeros, det_y_fit, residues, d_const = VF_lib.vector_fitting_zeros(
            s, det_y, n_order
        )

    # ---- BIC ----
    n_params = n_params_for_order(n_order)
    bic_val = compute_bic(det_y, det_y_fit, n_params)
    rmse = np.sqrt(np.mean(np.abs(det_y - det_y_fit) ** 2))

    modes = build_modes_table(zeros)

    analytical_expr = build_analytical_expression(poles, residues, d_const)

    if not silent:
        print(f"\n{'='*60}")
        print(f"  BEST FIT by BIC  —  order = {n_order}")
        print(f"{'='*60}")
        print(f"  RMSE       = {rmse:.6e}")
        print(f"  n_params   = {n_params}")
        print(f"  BIC        = {bic_val:.4f}  (lower is better)")
        print(f"\n  Analytical expression:")
        print(f"  f(s) = {analytical_expr}")
        print(f"\n  --- Oscillation Modes (zeros of det(Y_LIM)) ---")
        print(modes.to_string(index=False))

    # Always save text result.
    prefix = f"VF_fit_order{n_order}"
    txt_save_path = os.path.join(output_dir, f"{prefix}_TransferFunction_and_Modes.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f_out:
        f_out.write("====== Vector Fitting Analytical Expression ======\n\n")
        f_out.write(f"Order  : {n_order}\n")
        f_out.write(f"RMSE   : {rmse:.6e}\n")
        f_out.write(f"BIC    : {bic_val:.4f}\n\n")
        f_out.write(f"f(s) = {analytical_expr}\n\n")
        f_out.write("====== Modes Based on Zeros ======\n\n")
        f_out.write(modes.to_string(index=False))
    if not silent:
        print(f"\n[OK] Vector Fitting result saved to: {txt_save_path}")

    # Comparison plot — only for the best (non-silent) fit.
    if not silent:
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(freq, np.abs(det_y), "b", label=r"Original $|\det(Y)|$")
        plt.plot(freq, np.abs(det_y_fit), "r--", label="VF Fit")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(freq, np.angle(det_y), "b", label="Original Phase")
        plt.plot(freq, np.angle(det_y_fit), "r--", label="VF Fit")
        plt.ylabel("Phase (rad)")
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        png_save_path = os.path.join(output_dir, "VF_fit_BEST_plot.png")
        plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Vector Fitting comparison plot saved to: {png_save_path}\n")

        # s-plane pole-zero plot.
        plot_s_plane(poles, zeros, output_dir, n_order)

    return modes, bic_val, rmse


# =============================================================================
#  BIC sweep — fit with multiple orders and plot BIC vs. order
# =============================================================================

def bic_sweep(det_results_file, output_dir,
              min_order=2, max_order=20, step=1):
    """Fit det(Y_LIM) over a range of orders and plot BIC vs. order.

    Returns
    -------
    best_order : int   order with the smallest BIC
    bic_table  : pd.DataFrame
    """
    orders = list(range(min_order, max_order + 1, step))
    if USE_CONJUGATE_FITTING:
        orders = [n for n in orders if n % 2 == 0]
        if not orders:
            raise ValueError("No even orders in sweep range — "
                             "conjugate fitting requires even orders.")
    bic_list = []
    rmse_list = []

    print("\n" + "=" * 60)
    print("  BIC  Sweep  —  fitting det(Y_LIM) with orders "
          f"{min_order} … {max_order}")
    print("=" * 60)

    for n in orders:
        try:
            _, bic_val, rmse_val = fit_det_y_lim(det_results_file, output_dir,
                                                  n_order=n, silent=True)
        except Exception as exc:
            print(f"  [SKIP] order={n} failed: {exc}")
            bic_val, rmse_val = np.nan, np.nan
        bic_list.append(bic_val)
        rmse_list.append(rmse_val)
        print(f"  order={n:2d}  BIC={bic_val:12.4f}  RMSE={rmse_val:.6e}")

    bic_table = pd.DataFrame({
        "Order": orders,
        "n_params": [n_params_for_order(o) for o in orders],
        "RMSE": rmse_list,
        "BIC": bic_list,
    })
    print("\n" + bic_table.to_string(index=False))

    # ---- Plot BIC vs order ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   sharex=True)
    fig.suptitle("BIC Sweep — Model Order Selection", fontsize=14)

    valid = ~np.isnan(bic_list)
    ax1.plot(np.array(orders)[valid], np.array(bic_list)[valid],
             "bo-", linewidth=1.5)
    best_idx = np.nanargmin(bic_list)
    ax1.plot(orders[best_idx], bic_list[best_idx],
             "r*", markersize=15, label=f"Best N={orders[best_idx]}")
    ax1.set_ylabel("BIC")
    ax1.legend()
    ax1.grid(True, ls="--", alpha=0.7)

    ax2.plot(np.array(orders)[valid], np.array(rmse_list)[valid],
             "go-", linewidth=1.5)
    ax2.set_ylabel("RMSE")
    ax2.set_xlabel("Model Order N")
    ax2.grid(True, ls="--", alpha=0.7)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

    png_path = os.path.join(output_dir, "BIC_sweep.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\n[OK] BIC sweep plot saved to: {png_path}")

    # Save table.
    csv_path = os.path.join(output_dir, "BIC_sweep_results.csv")
    bic_table.to_csv(csv_path, index=False, float_format="%.6e")
    print(f"[OK] BIC sweep table saved to: {csv_path}")

    best_order = orders[best_idx]
    print(f"\n>>> Best order by BIC: N = {best_order}  "
          f"(BIC = {bic_list[best_idx]:.4f})")

    # ---- Re-fit with best order (show plot & modes) ----
    print("\n>>> Re-fitting with best order and generating plots …")
    fit_det_y_lim(det_results_file, output_dir,
                  n_order=best_order, silent=False)

    return best_order, bic_table


# =============================================================================
#  Main
# =============================================================================

def main():
    freq, y_lim, det_y_lim = calculate_y_lim(IMTB_DATA_FILE)

    det_results_file = save_det_y_lim(freq, det_y_lim, OUTPUT_DIR)

    plot_det_bode(freq, det_y_lim, OUTPUT_DIR)
    plot_bode_mimo_linear(freq, y_lim, OUTPUT_DIR)

    mag_dd = 20 * np.log10(np.abs(y_lim["dd"]))
    peak_idx = np.argmax(mag_dd)
    print(f"\nYdd peak frequency: {freq[peak_idx]:.2f} Hz")
    print(f"Ydd peak magnitude: {mag_dd[peak_idx]:.2f} dB")

    # ---- BIC sweep (automatically picks & plots the best order) ----
    print("\n>>> BIC sweep <<<")
    bic_sweep(det_results_file, OUTPUT_DIR,
              min_order=BIC_SWEEP_MIN_ORDER,
              max_order=BIC_SWEEP_MAX_ORDER,
              step=BIC_SWEEP_STEP)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
