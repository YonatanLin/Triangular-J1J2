import argparse

import numpy as np

from Trying2D import TriangularJ1J2DMRG, TriangularJ1J2CaseDirName
from ClusterInputConfigurations import dmrg_input_params, build_parser
from pathlib import Path

def main():
    args = build_parser(dmrg_input_params).parse_args()
    args_dict = vars(args)
    kwargs = {param: args_dict[param] for param, param_type in dmrg_input_params}
    TriangularJ1J2DMRG(**kwargs)


if __name__ == "__main__":
    main()
