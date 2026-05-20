import argparse

import numpy as np

from Trying2D import TriangularJ1J2DMRG, TriangularJ1J2CaseDirName
from TriangularJ1J2DMRGConfig import dmrg_input_params, parse_bool
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TestTriangularLattice with command line parameters."
    )
    for param, param_type in dmrg_input_params:
        parser.add_argument("--" + param, type=param_type, required=True)

    return parser

def main():
    args = build_parser().parse_args()
    args_dict = vars(args)
    kwargs = {param: args_dict[param] for param, param_type in dmrg_input_params}
    TriangularJ1J2DMRG(**kwargs)


if __name__ == "__main__":
    main()
