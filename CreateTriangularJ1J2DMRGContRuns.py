from Main import CreateTriangularCaseDir, DEFAULT_DELZ
from ClusterInputConfigurations import dmrg_input_params
import sys
from pathlib import Path
import re
import os


def AddTriangularCaseDirsToCondorCases(main_results_dir, condor_cases_file="condor_cases.txt", chi_input=None,
        max_sweeps_input=None):
    # Main.TriangularGeometryDirName: _rm_<remove_sites> after the geometry if sites are removed
    geometry_re = re.compile(r"^Lx_(.+)_Ly_(.+)_bc_([op]{2})_(YC|XC)(?:_rm_(.+))?$")
    params_re = re.compile(r"^(.+)_init_(.+)_conserve_(.+)_J2_([^_]+)(?:_Delz_([^_]+))?_chi_(.+)_maxsweeps_([^_]+)"
                           r"(_mix_0)?(_ramp_0)?(?:_magz_([^_]+))?(?:_.*)?$")

    with open(condor_cases_file, "w") as input_for_condor:
        for geometry_dir in sorted(Path(main_results_dir).iterdir()):
            if not geometry_dir.is_dir():
                continue

            geometry_match = geometry_re.match(geometry_dir.name)
            #print(geometry_match)
            if geometry_match is None:
                continue
            Lx, Ly, bc_short, geometry, remove_sites = geometry_match.groups()
            bc = "-".join("periodic" if ax == "p" else "open" for ax in bc_short)
            for params_dir in sorted(geometry_dir.iterdir()):
                if not params_dir.is_dir():
                    continue
                params_match = params_re.match(params_dir.name)
                #print(params_match)
                if params_match is None:
                    continue

                (bc_MPS, _initial_state, conserve, J2, Delz, chi, max_sweeps, no_mixer, no_ramp,
                 norm_magz) = params_match.groups()
                if norm_magz is None:
                    norm_magz = 0.0
                mixer = False if no_mixer else None  # the continuation keeps the mixer setting
                # the continuation starts from file at chi_max anyway; the setting only keeps its directory name
                ramp = False if no_ramp else None
                if Delz is None:
                    Delz = DEFAULT_DELZ  # Main.J1J2ParamsDirName leaves the default out
                if "cont" in _initial_state:
                    continue

                case_folder = str(params_dir).replace("\\", "/") + "/"
                cont_folder = str(case_folder).replace(_initial_state, f"cont_{_initial_state}")
                if chi_input is not None:
                    #print(f"replacing {chi} with ")
                    #print(cont_folder)
                    #print(f"replacing chi_{chi} with chi_{chi_input}")
                    cont_folder = str(cont_folder).replace(f"chi_{chi}", f"chi_{chi_input}")
                    chi = chi_input
                if max_sweeps_input is not None:
                    cont_folder = str(cont_folder).replace(f"maxsweeps_{max_sweeps}", f"maxsweeps_{max_sweeps_input}")
                    max_sweeps = max_sweeps_input
                Path(cont_folder).mkdir(parents=True, exist_ok=True)
                case_folder_absolute = os.getcwd() + "/" + case_folder
                row = {
                    "Lx": Lx,
                    "Ly": Ly,
                    "bc": bc,
                    "bc_MPS": bc_MPS,
                    "flux": "0.0",
                    "initial_state": "from_file",
                    "conserve": conserve,
                    "J2": J2,
                    "geometry": geometry,
                    "chi_max": chi,
                    "max_sweeps": max_sweeps,
                    "norm_magz": norm_magz,
                    "initial_psi_dir": case_folder_absolute,
                    "Delz": Delz,
                    "mixer": mixer,
                    "ramp": ramp,
                    "remove_sites": remove_sites,
                }
                input_for_condor.write(" ".join(str(row[field[0]]) for field in dmrg_input_params) + f" {cont_folder}\n")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        AddTriangularCaseDirsToCondorCases(sys.argv[1], chi_input=sys.argv[2])
    elif len(sys.argv) == 4:
        AddTriangularCaseDirsToCondorCases(sys.argv[1], chi_input=sys.argv[2], max_sweeps_input=sys.argv[3])
    else:
        AddTriangularCaseDirsToCondorCases(sys.argv[1])


