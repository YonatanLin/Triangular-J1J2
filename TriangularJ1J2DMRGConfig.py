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

params_excluded_from_dirname = ["initial_psi_dir"]

def dmrg_input_header():
    return " ".join(dmrg_input_params)

