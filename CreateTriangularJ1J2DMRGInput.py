from Trying2D import CreateTriangularCaseDir
import sys
from pathlib import Path
import re
import os 


def CreateTriangularCaseDirFromInputFile(main_results_dir, input_file):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
    params_names = input_file_lines[0].split(" ")
    print(params_names)
    assert(params_names[0] == "Lx" and params_names[1] == "Ly" and params_names[2] == "bc" and params_names[3] == "bc_MPS"
           and params_names[4] == "flux" and params_names[5] == "initial_state" and params_names[6] == "conserve" and
           params_names[7] == "J2" and params_names[8] == "geometry" and params_names[9] == "chi" and params_names[10] == "max_sweeps" and params_names[11]=="initial_psi_dir\n")
    input_for_condor = open("condor_cases.txt", 'w')
    for line in input_file_lines[1:]:
        params = line.split(" ")
        case_folder = CreateTriangularCaseDir(main_results_dir, params[0], params[1], params[2].split("-"),
                                              params[3], params[5], params[6], params[7], params[8], params[9], params[10])
        input_for_condor.write(line[:-1] + " " + case_folder + "\n")



if __name__ == "__main__":
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2])
