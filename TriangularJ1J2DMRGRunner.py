import argparse

import numpy as np

from Main import TriangularJ1J2DMRG, TriangularJ1J2CaseDirName
from ClusterInputConfigurations import dmrg_input_params, build_parser, input_param_name
from pathlib import Path

def main():
    args = build_parser(dmrg_input_params).parse_args()
    args_dict = vars(args)
    kwargs = {input_param_name(param): args_dict[input_param_name(param)] for param in dmrg_input_params}
    TriangularJ1J2DMRG(**kwargs)


if __name__ == "__main__":
    main()
