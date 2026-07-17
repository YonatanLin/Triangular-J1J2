from Trying2D import GutzwillerDMRGOverlaps
from ClusterInputConfigurations import build_parser, dmrg_gutz_comp_input_params
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    args = build_parser(dmrg_gutz_comp_input_params).parse_args()
    args_dict = vars(args)

    kwargs = {param: args_dict[param] for param, param_type in dmrg_input_params}

    J2_file = args_dict["J2_file"]
    print(f"input file: {J2_file}")
    J2s = np.loadtxt(J2_file)
    if(J2s.shape == ()):
        J2s = np.array([J2s])
    kwargs = {param: args_dict[param] for param, param_type in dmrg_gutz_comp_input_params
              if param != "J2_file"}
    GutzwillerDMRGOverlaps(J2s=J2s, output_dir="./", **kwargs)
