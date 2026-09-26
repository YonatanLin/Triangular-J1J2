import os
import numpy as np
from ClusterInputConfigurations import build_parser, afm_dispersion_input_params
from TestTyEffective import afm_excitation_energy


def main():
    args = build_parser(afm_dispersion_input_params).parse_args()
    E = afm_excitation_energy(args.Ly, args.chi_max, args.kx, args.ky)
    print(f'E(kx={args.kx}, ky={args.ky}) = {E:.8f}')
    savepath = os.path.join(os.getcwd(), f"E_kx_{args.kx:.4f}_ky_{args.ky:.4f}.dat")
    np.savetxt(savepath, [[args.kx / np.pi, args.ky / np.pi, E]], header='kx/pi   ky/pi   E')


if __name__ == "__main__":
    main()
