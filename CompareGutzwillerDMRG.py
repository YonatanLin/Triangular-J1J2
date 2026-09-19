from Main import GutzwillerDMRGOverlaps
from ClusterInputConfigurations import build_parser, dmrg_gutz_comp_input_params, input_param_name
import numpy as np
from pathlib import Path


def scanned_parameter_name(parameter_file, available_parameter_names):
    file_stem = Path(parameter_file).stem
    if file_stem in available_parameter_names:
        return file_stem
    if file_stem.endswith("s") and file_stem[:-1] in available_parameter_names:
        return file_stem[:-1]
    raise ValueError(f"Cannot infer scanned parameter from parameter file name: {parameter_file}")


if __name__ == "__main__":
    args = build_parser(dmrg_gutz_comp_input_params).parse_args()
    args_dict = vars(args)

    parameter_file = args_dict["parameter_file"]
    print(f"input file: {parameter_file}")
    kwargs = {input_param_name(param): args_dict[input_param_name(param)] for param in dmrg_gutz_comp_input_params
              if input_param_name(param) != "parameter_file"}

    scan_name = scanned_parameter_name(parameter_file, kwargs.keys())
    parameter_values = np.atleast_1d(np.loadtxt(parameter_file))
    GutzwillerDMRGOverlaps(scanned_parameter_name=scan_name, scanned_parameter_values=parameter_values,
                           output_dir="./", **kwargs)
