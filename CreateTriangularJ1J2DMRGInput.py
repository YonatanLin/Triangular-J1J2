from Trying2D import CreateTriangularCaseDir
from TriangularJ1J2DMRGConfig import dmrg_input_params, params_excluded_from_dirname
import sys


def CreateTriangularCaseDirFromInputFile(main_results_dir, input_file):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
    params_names = input_file_lines[0].strip().split(" ")
    expected_params = [param_data[0] for param_data in dmrg_input_params]
    assert params_names == expected_params, f"Bad input header: got {params_names}, expected {expected_params}"

    input_for_condor = open("condor_cases.txt", 'w')
    for line in input_file_lines[1:]:
        params = line.strip().split(" ")
        n_tokens = len(line.split(" "))
        expected_n_tokens = len(expected_params)
        assert(n_tokens == expected_n_tokens), f"line has {n_tokens} tokens, expected {expected_n_tokens} tokens"
        kwargs = {param[0]: params[i_param] for i_param, param in enumerate(dmrg_input_params)
                  if (param[0] not in params_excluded_from_dirname)}
        kwargs["bc"] = kwargs["bc"].split("-")
        case_folder = CreateTriangularCaseDir(main_results_dir, **kwargs)
        input_for_condor.write(line[:-1] + " " + case_folder + "\n")



if __name__ == "__main__":
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2])
