import argparse

def parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def optional_int(value):
    if value == "None":
        return None
    else:
        return int(value)

def optional_str(value):
    if value == "None":
        return None
    else:
        return str(value)


dmrg_input_params = [
    ("Lx", int),
    ("Ly", int),
    ("bc", str),
    ("bc_MPS", str),
    ("initial_state", str),
    ("conserve", parse_bool),
    ("J2", float),
    ("geometry", str),
    ("chi_max", int),
    ("max_sweeps", int),
    ("initial_psi_dir", str)]

dmrg_params_excluded_from_dirname = ["initial_psi_dir"]

gutzwiller_input_params = [
    ("Lx", int),
    ("Ly", int),
    ("chi_max", int),
    ("flux", float),
    ("geometry", str),
    ("bc_MPS", str),
    ("gs_manifold_index", int),
    ("model_type", optional_str)
]


dmrg_gutz_comp_input_params = [("Lx", int), ("Ly", int), ("geometry", str), ("bc_MPS", str),
                               ("dmrg_initial_state", str), ("dmrg_conserve", int),
                               ("dmrg_chi_max", optional_int), ("dmrg_max_sweeps", optional_int), ("dmrg_parent_dir", str),
                               ("gutz_chi_max", int), ("gutz_flux", float), ("gutz_gs_manifold_index", int),
                               ("gutz_parent_dir", str), ("J2_file", str)]

def build_parser(input_params):
    parser = argparse.ArgumentParser(
        description="Run TestTriangularLattice with command line parameters."
    )
    for param, param_type in input_params:
        parser.add_argument("--" + param, type=param_type, required=True)

    return parser


def CreateTriangularCaseDirFromInputFile(main_results_dir, input_file, input_params, excluded_params,
                                         dir_name_generator, condor_cases_filename):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
    params_names = input_file_lines[0].strip().split(" ")
    expected_params = [param_data[0] for param_data in input_params]
    assert params_names == expected_params, f"Bad input header: got {params_names}, expected {expected_params}"

    input_for_condor = open(condor_cases_filename, 'w')
    for line in input_file_lines[1:]:
        params = line.strip().split(" ")
        n_tokens = len(line.split(" "))
        expected_n_tokens = len(expected_params)
        assert(n_tokens == expected_n_tokens), f"line has {n_tokens} tokens, expected {expected_n_tokens} tokens"
        
        kwargs = {param[0]: params[i_param] for i_param, param in enumerate(input_params)
                  if (param[0] not in excluded_params)}
        if "bc" in kwargs:
            kwargs["bc"] = kwargs["bc"].split("-")
        
        case_folder = dir_name_generator(main_results_dir, **kwargs)
        input_for_condor.write(line[:-1] + " " + case_folder + "\n")
