import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import VF_lib


# Input files exported by AIM/PSCAD.
TERMINAL2_ADMITTANCE_FILE = (
r"testbed_8800MW_Type4&TG\Type4_X=0.3_ip\Measurementdata_Toolbox1\Terminal2\T2_admittance_dq_MIMO1.txt"
)
TERMINAL1_ADMITTANCE_FILE = (
r"testbed_8800MW_Type4&TG\Type4_X=0.3_ip\Measurementdata_Toolbox1\Terminal1\T1_admittance_dq_MIMO1.txt"
)

# All output files are written here.
OUTPUT_DIR = r"testbed_8800MW_Type4&TG\Type4_X=0.3_ip"

# Vector Fitting settings.
N_ORDER = 6
FIT_NROWS = 120
USE_CONJUGATE_FITTING = False
SHOW_PLOTS = True


def load_and_convert_to_complex_y(filename):
    """Read dq admittance data and convert dB/degree columns to complex values."""
    df = pd.read_csv(filename, sep=r"\s+")
    freq = df["fp"].values

    def to_complex(mag_db, pha_deg):
        mag_lin = 10 ** (mag_db / 20.0)
        pha_rad = np.deg2rad(pha_deg)
        return mag_lin * np.exp(1j * pha_rad)

    admittance = {
        "dd": to_complex(df["Ydd_mag"], df["Ydd_pha"]),
        "dq": to_complex(df["Ydq_mag"], df["Ydq_pha"]),
        "qd": to_complex(df["Yqd_mag"], df["Yqd_pha"]),
        "qq": to_complex(df["Yqq_mag"], df["Yqq_pha"]),
    }
    return freq, admittance


def calculate_y_lim(terminal2_file, terminal1_file):
    """Calculate Y_LIM and det(Y_LIM) from two-terminal admittance data."""
    freq, y_s = load_and_convert_to_complex_y(terminal2_file)
    freq_terminal1, y_g = load_and_convert_to_complex_y(terminal1_file)

    if not np.array_equal(freq, freq_terminal1):
        raise ValueError("The frequency columns of terminal1 and terminal2 do not match.")

    y_lim = {}
    for component in ["dd", "dq", "qd", "qq"]:
        y_lim[component] = y_s[component] + y_g[component]

    det_y_lim = y_lim["dd"] * y_lim["qq"] - y_lim["dq"] * y_lim["qd"]
    return freq, y_lim, det_y_lim


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


def fit_det_y_lim(det_results_file, output_dir):
    df = pd.read_csv(det_results_file, sep=r"\s+", nrows=FIT_NROWS)
    freq = df["Frequency_Hz"].values
    s = 1j * 2 * np.pi * freq
    det_y = df["Real_Part"].values + 1j * df["Imag_Part"].values

    if USE_CONJUGATE_FITTING:
        poles, zeros, det_y_fit, residues, d_const = VF_lib.vector_fitting_zeros_conjugate(
            s, det_y, N_ORDER
        )
    else:
        poles, zeros, det_y_fit, residues, d_const = VF_lib.vector_fitting_zeros(
            s, det_y, N_ORDER
        )

    modes = pd.DataFrame(
        {
            "Real_Sigma": zeros.real,
            "Imag_Omega": zeros.imag,
            "Freq_Hz": zeros.imag / (2 * np.pi),
            "Damping_Ratio_%": -zeros.real / np.abs(zeros) * 100,
        }
    ).sort_values("Freq_Hz")

    analytical_expr = build_analytical_expression(poles, residues, d_const)

    print("\n--- det(Y_LIM) fit analytical expression ---")
    print(f"f(s) = {analytical_expr}")
    print("\n--- Modes based on zeros ---")
    print(modes)

    txt_save_path = os.path.join(output_dir, "VF_fit_TransferFunction_and_Modes.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f_out:
        f_out.write("====== Vector Fitting Analytical Expression ======\n\n")
        f_out.write(f"f(s) = {analytical_expr}\n\n")
        f_out.write("====== Modes Based on Zeros ======\n\n")
        f_out.write(modes.to_string(index=False))

    print(f"\n[OK] Vector Fitting result saved to: {txt_save_path}")

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

    png_save_path = os.path.join(output_dir, "VF_fit_plot.png")
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Vector Fitting comparison plot saved to: {png_save_path}\n")

    return modes


def main():
    freq, y_lim, det_y_lim = calculate_y_lim(
        TERMINAL2_ADMITTANCE_FILE,
        TERMINAL1_ADMITTANCE_FILE,
    )
    det_results_file = save_det_y_lim(freq, det_y_lim, OUTPUT_DIR)

    plot_det_bode(freq, det_y_lim, OUTPUT_DIR)
    plot_bode_mimo_linear(freq, y_lim, OUTPUT_DIR)

    mag_dd = 20 * np.log10(np.abs(y_lim["dd"]))
    peak_idx = np.argmax(mag_dd)
    print(f"Ydd peak frequency: {freq[peak_idx]:.2f} Hz")
    print(f"Ydd peak magnitude: {mag_dd[peak_idx]:.2f} dB")

    fit_det_y_lim(det_results_file, OUTPUT_DIR)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
