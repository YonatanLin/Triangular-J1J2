from Trying2D import CreateTriangularCaseDir
import sys
from pathlib import Path
import re
import os


def AddTriangularCaseDirsToCondorCases(main_results_dir, condor_cases_file="condor_cases.txt", chi_input=None):
    geometry_re = re.compile(r"^Lx_(.+)_Ly_(.+)_bc_([op]{2})_(.+)$")
    params_re = re.compile(r"^(.+)_init_(.+)_conserve_(.+)_J2_(.+)_chi_(.+)_maxsweeps_(.+)$")

    with open(condor_cases_file, "w") as input_for_condor:
        for geometry_dir in sorted(Path(main_results_dir).iterdir()):
            if not geometry_dir.is_dir():
                continue

            geometry_match = geometry_re.match(geometry_dir.name)
            #print(geometry_match)
            if geometry_match is None:
                continue
            Lx, Ly, bc_short, geometry = geometry_match.groups()
            bc = "-".join("periodic" if ax == "p" else "open" for ax in bc_short)
            for params_dir in sorted(geometry_dir.iterdir()):
                if not params_dir.is_dir():
                    continue
                params_match = params_re.match(params_dir.name)
                #print(params_match)
                if params_match is None:
                    continue

                bc_MPS, _initial_state, conserve, J2, chi, max_sweeps = params_match.groups()
                if "cont" in _initial_state:
                    continue

                case_folder = str(params_dir).replace("\\", "/") + "/"
                cont_folder = str(case_folder).replace(_initial_state, f"cont_{_initial_state}")
                if chi_input is not None:
                    #print(f"replacing {chi} with ")
                    cont_folder = str(cont_folder).replace(chi, f"{chi_input}")
                    chi = chi_input
                Path(cont_folder).mkdir(parents=True, exist_ok=True)
                case_folder_absolute = os.getcwd() + "/" + case_folder
                input_for_condor.write(
                    f"{Lx} {Ly} {bc} {bc_MPS} 0.0 from_file {conserve} {J2} {geometry} {chi} {max_sweeps} {case_folder_absolute} {cont_folder}\n"
                )

if __name__ == "__main__":
    AddTriangularCaseDirsToCondorCases(sys.argv[1])

