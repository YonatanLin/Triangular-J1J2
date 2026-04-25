import argparse

from Trying2D import TryPiFluxMonopoleState

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TryPiFluxMonopoleState with command line parameters."
    )
    parser.add_argument("--Lx", type=int, required=True, help="Number of unit cells along x.")
    parser.add_argument("--Ly", type=int, required=True, help="Number of unit cells along y.")
    parser.add_argument("--chi_max", type=int, default=1000, help="Max bond dimension.")
    parser.add_argument("--monopole_Q", type=int, default=1, help="Monopole charge.")
    parser.add_argument("--magnetization", type=float, default=0.0, help="Total magnetization.")

    return parser

def main():
    args = build_parser().parse_args()
    TryPiFluxMonopoleState(
        Lx=args.Lx,
        Ly=args.Ly,
        chi_max=args.chi_max,
        monopole_Q=args.monopole_Q,
        magnetization=args.magnetization,
    )


if __name__ == "__main__":
    main()
