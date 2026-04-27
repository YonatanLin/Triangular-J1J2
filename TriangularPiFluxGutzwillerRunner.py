import argparse

from numpy import float128

from Trying2D import TriangularPiFluxGutzwiller


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TriangularPiFluxGutzwiller with command line parameters."
    )
    parser.add_argument("--Lx", type=int, required=True, help="Number of unit cells along x.")
    parser.add_argument("--Ly", type=int, required=True, help="Number of unit cells along y.")
    parser.add_argument("--chi_max", type=int, required=True, help="Max bond dimension.")
    parser.add_argument("--flux", type=float, required=True, help="Net flux threaded through the cylinder")
    return parser


def main():
    args = build_parser().parse_args()
    TriangularPiFluxGutzwiller(Lx=args.Lx, Ly=args.Ly, chi_max=args.chi_max, flux=args.flux)


if __name__ == "__main__":
    main()
