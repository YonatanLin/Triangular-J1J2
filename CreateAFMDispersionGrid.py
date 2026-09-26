import sys
from pathlib import Path
import numpy as np


def main():
    """python CreateAFMDispersionGrid.py main_results_dir Ly chi_max [n_kx]
    kx: n_kx points on [0, pi]; ky = 2 pi n / Ly, n = 0 .. Ly-1. Writes afm_dispersion_condor_cases.txt."""
    main_results_dir, Ly, chi_max = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n_kx = int(sys.argv[4]) if len(sys.argv) > 4 else 13
    assert main_results_dir[-1] == "/"
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    kxs = np.linspace(0, np.pi, n_kx)
    kys = 2 * np.pi * np.arange(Ly) / Ly
    with open("afm_dispersion_condor_cases.txt", "w") as f:
        for kx in kxs:
            for ky in kys:
                case_dir = main_results_dir + f"AFM_dispersion_Ly_{Ly}_chi_{chi_max}_kx_{kx:.4f}_ky_{ky:.4f}/"
                Path(case_dir).mkdir(parents=True, exist_ok=True)
                f.write(f"{Ly} {chi_max} {kx:.10f} {ky:.10f} {case_dir}\n")


if __name__ == "__main__":
    main()
