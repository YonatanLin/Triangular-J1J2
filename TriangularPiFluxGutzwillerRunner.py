import argparse

from Trying2D import TriangularPiFluxGutzwiller

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TriangularPiFluxGutzwiller with command line parameters."
    )
    parser.add_argument("--Lx", type=int, required=True, help="Number of unit cells along x.")
    parser.add_argument("--Ly", type=int, required=True, help="Number of unit cells along y.")
    parser.add_argument("--chi_max", type=int, required=True, help="Max bond dimension.")

    return parser

def main():
    args = build_parser().parse_args()
    TriangularPiFluxGutzwiller(Lx=args.Lx, Ly=args.Ly, chi_max=args.chi_max)


if __name__ == "__main__":
    main()
