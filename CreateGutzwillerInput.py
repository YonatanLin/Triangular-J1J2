from pathlib import Path
import sys
from Trying2D import CreateGutzwillerCaseDir


def CreateGutzwillerCaseDirFromInputFile(main_results_dir, input_file):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
    params = input_file_lines[0].split(" ")
    print(params)
    assert(params[0] == "Lx" and params[1] == "Ly" and params[2] == "chi" and params[3] == "flux" and
           params[4] == "geometry\n")
    input_for_condor = open("gutz_condor_cases.txt", 'w')
    for line in input_file_lines[1:]:
        params = line.split(" ")
        case_folder = CreateGutzwillerCaseDir(main_results_dir, params[0], params[1], params[2],
                                              params[3], params[4].split("\n")[0])
        input_for_condor.write(line[:-1] + " " + case_folder + "\n")


if __name__ == "__main__":
    assert(sys.argv[1][-1] == "/")
    CreateGutzwillerCaseDirFromInputFile(sys.argv[1], sys.argv[2])