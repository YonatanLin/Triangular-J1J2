from Trying2D import GutzwillerDMRGOverlaps
from ClusterInputConfigurations import build_parser, dmrg_gutz_comp_input_params
import numpy as np


if __name__ == "__main__":
    # J2s = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.125, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    #J2s = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.13, 0.15, 0.17, 0.19, 0.2]
    # J2s = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.125, 0.13, 0.14, 0.15, 0.17, 0.2]
    #J2s = [0.125]
    #chi_gutz = 8000
    #chi_gutz = 15000
    #flux_gutz = 0.0
    #gutz_dir = "../PiFluxGutzwiller_79c67f5c1652485/"
    # gutz_dir = "../PiFluxGutzwiller_5_4__5135aa/"
    #gutz_dir = "../PiFluxGutzwiller_5_8_b9e663/"
    #Lx_dmrg, Ly = 2, 6
    #Lx_gutz = 80
    #dmrg_parent_dir = "../TriangularJ1J2DMRG_2c96287c0a582/"
    # dmrg_parent_dir = "../TriangularJ1J2DMRG_49de97dff33dfbb/"
    # dmrg_parent_dir = "../TriangularJ1J2DMRG_79c67f5c16524/"
    #dmrg_parent_dir = "../TriangularJ1J2DMRG_5_4__5135aa/"
    # dmrg_parent_dir = "../TriangularJ1J2DMRG_5_8_b9e663/"
    #geometry = "YC"
    #dmrg_initial_state="cont_Random"
    #bc_MPS = "infinite"
    #gs_manifold_index = 0

    args = build_parser(dmrg_gutz_comp_input_params).parse_args()
    args_dict = vars(args)

    output_dir = ""
    for param in args_dict.keys():
        if param != "J2_file":
            output_dir += f"{param}_{args_dict[param]}"


    #Lx = args_dict["Lx"]
    #Ly = args_dict["Ly"]
    #geometry = args_dict["geometry"]
    #bc_MPS = args_dict["bc_MPS"]
    #dmrg_initial_state = args_dict["dmrg_initial_state"]
    #dmrg_chi = args_dict["dmrg_chi_max"]
    #gutz_chi = args_dict["gutz_chi_max"]
    #gutz_flux = args_dict["gutz_flux"]
    #dmrg_dir = args_dict["dmrg_parent_dir"]
    #gutz_dir = args_dict["gutz_dir"]


    J2_file = args_dict["J2_file"]
    J2s = np.loadtxt(J2_file)

    kwargs = {param: args_dict[param] for param, param_type in dmrg_gutz_comp_input_params
              if param != "J2_file"}

    #GutzwillerDMRGOverlaps(J2s, args.gutz_dir, args.Lx_dmrg, args.Lx_gutz, args.Ly, args.chi_gutz,
    #                       args.flux_gutz,
    #                       output_dir, args.dmrg_initial_state,
    #                       args.dmrg_parent_dir, args.geometry, args.bc_MPS, args.gs_manifold_index,
    #                       args.chi_dmrg, args.max_sweeps_dmrg)
    GutzwillerDMRGOverlaps(J2s, output_dir=output_dir, **kwargs)
