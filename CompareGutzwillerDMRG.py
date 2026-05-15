from Trying2D import GutzwillerDMRGOverlaps
import argparse

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TriangularPiFluxGutzwiller with command line parameters."
    )
    parser.add_argument("--Lx_dmrg", type=int, required=True)
    parser.add_argument("--Lx_gutz", type=int, required=True)
    parser.add_argument("--Ly", type=int, required=True)
    parser.add_argument("--J2", type=float, required=True)
    parser.add_argument("--gutz_dir", type=str, required=True)
    parser.add_argument("--chi_gutz", type=int, required=True)
    parser.add_argument("--flux_gutz", type=float, required=True)
    parser.add_argument("--dmrg_initial_state", type=str, required=True)
    parser.add_argument("--dmrg_parent_dir", type=str, required=True)
    parser.add_argument("--geometry", type=str, required=True)
    parser.add_argument("--bc_MPS", type=str, required=True)
    parser.add_argument("--gs_manifold_index", type=int, required=True)
    return parser


if __name__ == "__main__":
    output_dir = "./"
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
    
    args = build_parser().parse_args()
    
    GutzwillerDMRGOverlaps([args.J2], args.gutz_dir, args.Lx_dmrg, args.Lx_gutz, args.Ly, args.chi_gutz, 
                           args.flux_gutz,
                           output_dir, args.dmrg_initial_state, 
                           args.dmrg_parent_dir, args.geometry, args.bc_MPS, args.gs_manifold_index)
