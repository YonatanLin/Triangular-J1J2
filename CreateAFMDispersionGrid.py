import argparse
import os
from pathlib import Path
import numpy as np


def main():
    """Writes afm_dispersion_condor_cases.txt (columns: Ly chi_max kx ky gs_path case_folder).
    kx: n_kx points on [0, pi]; ky = 2 pi n / Ly, n = 0 .. Ly-1.
    Ground state: --gs_path if given (must exist); otherwise
    main_results_dir/AFM_groundstate_Ly_{Ly}_chi_{chi}/groundstate.pkl. If it does not exist, a single
    ground-state-only line (kx = ky = None) is written; rerun this script once that job is done."""
    parser = argparse.ArgumentParser()
    parser.add_argument("main_results_dir")
    parser.add_argument("Ly", type=int)
    parser.add_argument("chi_max", type=int)
    parser.add_argument("n_kx", type=int, nargs="?", default=13)
    parser.add_argument("--gs_path", default=None)
    args = parser.parse_args()
    Ly, chi_max, main_results_dir = args.Ly, args.chi_max, args.main_results_dir
    assert main_results_dir[-1] == "/"
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)

    if args.gs_path is not None:
        assert os.path.isfile(args.gs_path), f"ground state file {args.gs_path} not found"
        gs_path = args.gs_path
    else:
        gs_dir = main_results_dir + f"AFM_groundstate_Ly_{Ly}_chi_{chi_max}/"
        gs_path = gs_dir + "groundstate.pkl"

    with open("afm_dispersion_condor_cases.txt", "w") as f:
        if not os.path.isfile(gs_path):
            Path(gs_dir).mkdir(parents=True, exist_ok=True)
            f.write(f"{Ly} {chi_max} None None {gs_path} {gs_dir}\n")
            print(f"no ground state at {gs_path}: wrote a ground-state-only case; rerun after it finishes")
            return
        for kx in np.linspace(0, np.pi, args.n_kx):
            for ky in 2 * np.pi * np.arange(Ly) / Ly:
                case_dir = main_results_dir + f"AFM_dispersion_Ly_{Ly}_chi_{chi_max}_kx_{kx:.4f}_ky_{ky:.4f}/"
                Path(case_dir).mkdir(parents=True, exist_ok=True)
                f.write(f"{Ly} {chi_max} {kx:.10f} {ky:.10f} {gs_path} {case_dir}\n")


if __name__ == "__main__":
    main()
