import sys

from ClusterInputConfigurations import postprocess_input_params
from PostProcess import SUPPORTED_POST_PROCESS_TYPES


def _validate_post_process_type(post_process_type):
    if post_process_type not in SUPPORTED_POST_PROCESS_TYPES:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )


def CreatePostProcessInput(results_dir, post_process_type, condor_cases_filename="postprocess_condor_cases.txt"):
    _validate_post_process_type(post_process_type)
    with open(condor_cases_filename, "w") as input_for_condor:
        input_for_condor.write(f"{results_dir} {post_process_type}\n")


def CreatePostProcessInputFromFile(input_file, condor_cases_filename="postprocess_condor_cases.txt"):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()

    params_names = input_file_lines[0].strip().split(" ")
    expected_params = [param_data[0] for param_data in postprocess_input_params]
    assert params_names == expected_params, f"Bad input header: got {params_names}, expected {expected_params}"

    with open(condor_cases_filename, "w") as input_for_condor:
        for line in input_file_lines[1:]:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            params = stripped_line.split(" ")
            n_tokens = len(params)
            expected_n_tokens = len(expected_params)
            assert n_tokens == expected_n_tokens, f"line has {n_tokens} tokens, expected {expected_n_tokens} tokens"
            _validate_post_process_type(params[1])

            input_for_condor.write(stripped_line + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        CreatePostProcessInputFromFile(sys.argv[1])
    elif len(sys.argv) == 3:
        CreatePostProcessInput(sys.argv[1], sys.argv[2])
    else:
        raise ValueError("Usage: CreatePostProcessInput.py <input_file> OR <results_dir> <post_process_type>")
