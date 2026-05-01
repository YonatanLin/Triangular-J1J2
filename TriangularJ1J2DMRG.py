import argparse

import numpy as np

from Trying2D import TriangularJ1J2DMRG, TriangularJ1J2CaseDirName
from pathlib import Path

def parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TestTriangularLattice with command line parameters."
    )
    parser.add_argument("--Lx", type=int, required=True, help="Number of unit cells along x.")
    parser.add_argument("--Ly", type=int, required=True, help="Number of unit cells along y.")
    parser.add_argument(
        "--bc",
        type=str,
        required=True,
        help="Boundary condition for lattice",
    )
    parser.add_argument(
        "--bc_MPS",
        type=str,
        required=True,
        help="MPS boundary condition: 'finite', 'infinite'",
    )

    parser.add_argument("--flux", type=float, default=0.0, required=True,
                        help="Flux value. Only 0 is currently supported.")
    parser.add_argument("--J2", type=float, default=0.0, required=True, help="J2 - nnn coupling.")
    parser.add_argument(
        "--conserve",
        type=parse_bool,
        default=True,
        required=True,
        help="Whether to conserve Sz. Accepted values: true/false.",
    )
    parser.add_argument(
        "--initial_state",
        type=str,
        default="Random",
        required=True,
        help="Initial state for DMRG.",
    )
    parser.add_argument("--geometry", type=str, default="YC",
                        required=True, help="Geometry type - either XC or YC")

    parser.add_argument("--initial_psi_dir", type=str, default=None, required=True,
                        help="directory containing initial state for dmrg")

    return parser

def main():
    args = build_parser().parse_args()
    TriangularJ1J2DMRG(Lx=args.Lx, Ly=args.Ly, bc=args.bc, bc_MPS=args.bc_MPS,
                       conserve=args.conserve, initial_state=args.initial_state,
                       J2=args.J2, geometry=args.geometry, initial_psi_dir=args.initial_psi_dir)


if __name__ == "__main__":
    main()
