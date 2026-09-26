import os
import numpy as np
from ClusterInputConfigurations import build_parser, afm_dispersion_input_params
from TestTyEffective import afm_excitation_energy, compute_and_save_afm_groundstate


def main():
    args = build_parser(afm_dispersion_input_params).parse_args()
    if args.kx is None or args.ky is None:
        compute_and_save_afm_groundstate(args.Ly, args.chi_max, args.gs_path)
        return
    E = afm_excitation_energy(args.Ly, args.kx, args.ky, args.gs_path)
    print(f'E(kx={args.kx}, ky={args.ky}) = {E:.8f}')
    savepath = os.path.join(os.getcwd(), f"E_kx_{args.kx:.4f}_ky_{args.ky:.4f}.dat")
    np.savetxt(savepath, [[args.kx / np.pi, args.ky / np.pi, E]], header='kx/pi   ky/pi   E')


if __name__ == "__main__":
    main()
