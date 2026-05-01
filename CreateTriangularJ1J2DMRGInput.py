from Trying2D import CreateTriangularCaseDir
import sys
from pathlib import Path
import re

def CreateTriangularCaseDirFromInputFile(main_results_dir, input_file):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
    params_names = input_file_lines[0].split(" ")
    print(params_names)
    assert(params_names[0] == "Lx" and params_names[1] == "Ly" and params_names[2] == "bc" and params_names[3] == "bc_MPS"
           and params_names[4] == "flux" and params_names[5] == "initial_state" and params_names[6] == "conserve" and
           params_names[7] == "J2" and params_names[8] == "geometry" and params_names[9]=="initial_psi_dir\n")
    input_for_condor = open("condor_cases.txt", 'w')
    for line in input_file_lines[1:]:
        params = line.split(" ")
        case_folder = CreateTriangularCaseDir(main_results_dir, params[0], params[1], params[2].split("-"),
                                              params[3], params[5], params[6], params[7], params[8])
        input_for_condor.write(line[:-1] + " " + case_folder + "\n")


def AddTriangularCaseDirsToCondorCases(main_results_dir, condor_cases_file="condor_cases.txt"):
    geometry_re = re.compile(r"^Lx_(.+)_Ly_(.+)_bc_([op]{2})_(.+)$")
    params_re = re.compile(r"^(.+)_init_(.+)_conserve_(.+)_J2_(.+)$")

    with open(condor_cases_file, "w") as input_for_condor:
        for geometry_dir in sorted(Path(main_results_dir).iterdir()):
            if not geometry_dir.is_dir():
                continue

            geometry_match = geometry_re.match(geometry_dir.name)
            print(geometry_match)
            if geometry_match is None:
                continue
            Lx, Ly, bc_short, geometry = geometry_match.groups()
            bc = "-".join("periodic" if ax == "p" else "open" for ax in bc_short)
            for params_dir in sorted(geometry_dir.iterdir()):
                if not params_dir.is_dir():
                    continue
                params_match = params_re.match(params_dir.name) 
                print(params_match)
                if params_match is None:
                    continue

                bc_MPS, _initial_state, conserve, J2 = params_match.groups()
                if "cont" in _initial_state:
                    continue

                case_folder = str(params_dir).replace("\\", "/") + "/"
                cont_folder = str(case_folder).replace(_initial_state, f"cont_{_initial_state}")
                Path(cont_folder).mkdir(parents=True, exist_ok=True)
                input_for_condor.write(
                    f"{Lx} {Ly} {bc} {bc_MPS} 0.0 from_file {conserve} {J2} {geometry} {case_folder} {cont_folder}\n"
                )


if __name__ == "__main__":
    if len(sys.argv) == 3:
        CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        AddTriangularCaseDirsToCondorCases(sys.argv[1])
    else:
        raise ValueError("Usage: python CreateTriangularJ1J2DMRGInput.py <main_results_dir> [input_file]")
