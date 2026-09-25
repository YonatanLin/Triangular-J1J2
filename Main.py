import os.path

from TryingTemfpy import local
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from TryingTemfpy import rc_params
plt.rcParams.update(rc_params)

import tenpy
import numpy as np
from numpy import sin, cos, sqrt, pi
from tenpy.models.model import CouplingModel, CouplingMPOModel
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
import tenpy.models
from tenpy.algorithms import dmrg
from numpy.linalg import eigh, det, inv
from tenpy.tools.misc import setup_logging
import pickle
from tenpy.networks.site import FermionSite, SpinHalfSite
from tenpy.models.spins import SpinModel
from pathlib import Path
import json
from tenpy import networks
from tenpy import MPSEnvironment
import threading
import time
from WaveFunctionProperties import (plot_scalar_spin_chirality, compute_structure_factor_grid,
                                    plot_structure_factor, structure_factor, ComputeMomentumSpaceStructureFactor,
                                    CalculateSpinSpinCorrelations)

setup_logging(to_stdout="INFO")
if not local:
    print(f"num threads: {tenpy.tools.process.mkl_get_nthreads()}")


default_chi_max = 3000
default_dmrg_params = {'mixer': True, 'max_E_err': 1.0e-10, 'trunc_params': {'chi_max': default_chi_max, 'svd_min': 1.0e-7},
                    'combine': True, 'chi_list': {0: 50, 3: 100, 7: default_chi_max}, 'min_sweeps': 7, 'max_sweeps': 8,
                   'N_sweeps_check': 1}

project_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/"
code_dir = project_dir + "Code/"
glob_results_dir = project_dir + "Results/"

meetings_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Meetings/"


model_type_dirac = "Dirac"
model_type_Z2 = "Z2"

pauli_x = np.array([[0., 1.], [1., 0.]])
pauli_y = np.array([[0., -1j], [1j, 0.]])
pauli_z = np.array([[1., 0.0], [0.0, -1.0]])

def AbsMagzFromNormMagz(norm_magz, N_sites):
    print(f"norm_magz: {norm_magz}, N_sites: {N_sites}")
    magz_tot_doubled = int(round(norm_magz * N_sites))
    assert(magz_tot_doubled % 2 == 0)
    return magz_tot_doubled // 2


def ChangeChiInDMRGParams(dmrg_params, chi_max):
    dmrg_params["trunc_params"]["chi_max"] = chi_max
    dmrg_params["chi_list"] = {0: 50, 3: 100, 7: chi_max}


def CreateGutzwillerCaseDir(main_results_dir, Lx, Ly, chi_max, flux, geometry, bc_MPS,
                            gs_manifold_index, model_type, norm_magz, monopole_Q, svd_min=None):
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    case_name = f"{bc_MPS}_Lx_{Lx}_Ly_{Ly}_chi_{chi_max}_flux_{flux}_{geometry}_gsindex_{gs_manifold_index}"

    if model_type is not None:
        case_name = f"{model_type}_" + case_name
    if float(norm_magz) > 1e-15:
        norm_magz_float = float(norm_magz)
        case_name += f"_magz_{norm_magz_float:.4f}"
    if monopole_Q is not None:
        case_name += f"_monQ_{monopole_Q}"
    if svd_min is not None and str(svd_min) != "None":
        case_name += f"_svdmin_{svd_min}"

    case_name += "/"
    gutz_dir = main_results_dir + case_name
    Path(gutz_dir).mkdir(parents=True, exist_ok=True)
    return gutz_dir


def CreateOverlapsCaseDir(main_results_dir, **kwargs):
    Lx = kwargs.get("Lx")
    Ly = kwargs.get("Ly")
    gutz_chi_max = kwargs.get("gutz_chi_max")
    gutz_flux = kwargs.get("gutz_flux")
    gutz_mon_Q = kwargs.get("gutz_mon_Q")
    dmrg_initial_state = kwargs.get("dmrg_initial_state")
    geometry = kwargs.get("geometry")
    bc_MPS = kwargs.get("bc_MPS")
    dmrg_chi_max = kwargs.get("dmrg_chi_max")
    dmrg_max_sweeps = kwargs.get("dmrg_max_sweeps")
    norm_magz = kwargs.get("norm_magz")
    J2 = kwargs.get("J2")
    Delz = kwargs.get("Delz")
    parameter_file = kwargs.get("parameter_file")

    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    geometry_case_dir = f"{bc_MPS}_Lx_{Lx}_Ly_{Ly}_{geometry}/"
    overlaps_dir = main_results_dir + geometry_case_dir
    Path(overlaps_dir).mkdir(parents=True, exist_ok=True)
    scanned_parameter_name = None
    if parameter_file is not None:
        scanned_parameter_name = Path(parameter_file).stem
        if scanned_parameter_name.endswith("s") and scanned_parameter_name[:-1] in {"J2", "Delz"}:
            scanned_parameter_name = scanned_parameter_name[:-1]
    scan_dir = f"scan_{scanned_parameter_name}_" if scanned_parameter_name is not None else ""
    fixed_params_dir = ""
    if scanned_parameter_name != "J2" and J2 is not None:
        fixed_params_dir += f"J2_{J2}_"
    if scanned_parameter_name != "Delz" and Delz is not None and abs(float(Delz) - 1.0) > 1e-15:
        fixed_params_dir += f"Delz_{Delz}_"
    hamiltonian_case_dir = (f"chiGutz_{gutz_chi_max}_flux_{gutz_flux}_monQ_{gutz_mon_Q}_"
                              f"{scan_dir}{fixed_params_dir}"
                              f"initDMRG_{dmrg_initial_state}_chiDMRG_{dmrg_chi_max}_sweeps_{dmrg_max_sweeps}_"
                              f"magz_{norm_magz}/")
    overlaps_dir = overlaps_dir + hamiltonian_case_dir
    Path(overlaps_dir).mkdir(parents=True, exist_ok=True)
    return overlaps_dir


def AddAndTrackCoupling(model, strength, u1, op1, u2, op2, dx, couplings_list, plus_hc=False,
                        flux=0.0):
    if abs(flux) > 1e-15:
        assert(plus_hc)
        assert(model.lat.boundary_conditions[1] == "periodic") #should be a cylinder which is periodic in the y direction
        strength_with_flux = model.coupling_strength_add_ext_flux(strength, dx, [0, flux])
        model.add_coupling(strength_with_flux, u1, op1, u2, op2, dx, plus_hc=plus_hc)
    else:
        model.add_coupling(strength, u1, op1, u2, op2, dx, plus_hc=plus_hc)

    couplings_list.append((u1, u2, dx))


def PrintCouplings(model, include_sites=None):
    couplings_list = model.all_coupling_terms().to_TermList()
    couplings_dict = {}
    for coupling in couplings_list:
        site_i = coupling[0][0][1]
        site_j = coupling[0][1][1]
        coupling_strength = coupling[1]
        operators = (coupling[0][0][0], coupling[0][1][0])

        if include_sites is None:
            print(f"i={site_i}, j={site_j}: tet={np.angle(coupling_strength) / pi}, {operators[0]}, {operators[1]}")
        else:
            if(site_i in include_sites or site_j in include_sites):
                dict_key = (site_i, site_j, operators[0], operators[1])
                assert (dict_key not in couplings_dict.keys())
                couplings_dict[dict_key] = coupling_strength
                # print(f"i={site_i}, j={site_j}: {coupling_strength}, {operators[0]}, {operators[1]}")
    return couplings_dict


def getSpecielBzPoints():
    M1 = np.array([0., 2 * pi / sqrt(3)])
    M2 = np.array([pi, (-1) * pi / sqrt(3)])
    M3 = np.array([pi, pi / sqrt(3)])
    K = 2 * pi * np.array([1./3., 1./sqrt(3.)])
    K_prime = 2 * pi * np.array([-1./3., 1./sqrt(3.)])
    return {"M1": M1, "M2": M2, "M3": M3, "K": K, "K_prime": K_prime}


def getKPointName(k_2D):
    special_k_points = getSpecielBzPoints()
    for key in special_k_points.keys():
        if(np.linalg.norm(k_2D - special_k_points[key]) < 1e-14):
            return key
    return None



def PlotLattice(lat, ax, additional_couplings_to_plot=None, plot_nn_couplings=True, nnn_color="green",
                nnn_line_style="-", plot_order=True):
    #if add_nn_explicitly:
    #    lat.plot_coupling(ax, coupling=nn_couplings_list, linewidth=1.0)
    if plot_nn_couplings:
        lat.plot_coupling(ax, linewidth=1.0)

    if additional_couplings_to_plot is not None:
        lat.plot_coupling(ax, coupling=additional_couplings_to_plot, linewidth=0.5,
                      color="green", linestyle=nnn_line_style, wrap=True)
    if plot_order:
        lat.plot_order(ax)
    lat.plot_sites(ax)
    lat.plot_basis(ax, origin=-0.5 * (lat.basis[0] + lat.basis[1]))
    ax.set_aspect('equal')
    ax.set_xlim(-1)
    ax.set_ylim(-1)


def ImshowMatrix(ax, fig, X, Y, spin_corr_k, xlabel = r"$k_x$",
                ylabel=r"$k_y$", title=None, label=""):
    # if xlabel == r"$k_x / \pi$":
    #    norm = 1. / pi
    image = ax.imshow(
        np.real(spin_corr_k),
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='RdBu',
        aspect='auto', label=label
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$S(\mathbf{k})$")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color='white', linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color='white', linewidth=0.6, alpha=0.5)


def PlotSquareLatticeStructureFactor(Lx=6, Ly=6):
    site = SpinHalfSite(conserve=None)
    square_lat = tenpy.models.lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=['open', 'open'])

    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    square_lat.plot_order(ax_lat)
    square_lat.plot_coupling(ax_lat)
    square_lat.plot_sites(ax_lat)
    square_lat.plot_basis(ax_lat)

    product_state = []
    for i in range(square_lat.N_sites):
            lat_ind = square_lat.mps2lat_idx(i)
            x = lat_ind[0]
            y = lat_ind[1]
            product_state.append("up" if (x + y) % 2 == 0 else "down")

    psi = MPS.from_product_state(
        square_lat.mps_sites(),
        product_state,
        bc=square_lat.bc_MPS,
        unit_cell_width=square_lat.mps_unit_cell_width,
    )
    psi.canonical_form()

    spin_corr_x = CalculateSpinSpinCorrelations(psi)
    ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, square_lat)

    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
    ax.set_title(title)
    plot_structure_factor(ks, spin_corr_k, square_lat, ax)
    square_lat.plot_brillouin_zone(ax)

    fig.tight_layout()
    if local:
            plt.show()


def Generate120DegOrderedState(lat=None, plot=False):
    if lat == None:
        site = SpinHalfSite(conserve=None)
        Lx = 9
        Ly = 9
        triangular_lat = tenpy.models.lattice.Triangular(Lx=Lx, Ly=Ly, site=site, bc=['periodic', 'open'])
        lat = triangular_lat
    else:
        Lx, Ly = lat.Ls[0], lat.Ls[1]
        site = lat.mps_sites()[0]
    basis = lat.basis
    unitcell_pos = lat.unit_cell_positions

    aligned_with_x = (basis[0][1] == 0.0 or basis[1][1] == 0.0)
    aligned_with_y = (basis[0][0] == 0.0 or basis[1][0] == 0.0)
    assert(aligned_with_x or aligned_with_y)

    psi = MPS.from_product_state(
        lat.mps_sites(),
        ["up"] * lat.N_sites,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )

    rot_120_angle = 2 * pi / 3
    debug = abs(rot_120_angle - pi) < 1e-15
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    I2 = np.eye(2)
    rotation_op_120 = cos(rot_120_angle / 2.) * I2 - 1j * sin(rot_120_angle / 2.) * pauli_x
    rotation_op_240 = cos(rot_120_angle) * I2 - 1j * sin(rot_120_angle) * pauli_x
    site.add_op("rot_once", rotation_op_120)
    site.add_op("rot_twice", rotation_op_240)

    for i in range(lat.N_sites):
        lat_ind = lat.mps2lat_idx(i)
        if aligned_with_x:
            column_cor = lat_ind[0] * basis[0][0] + lat_ind[1] * basis[1][0] + unitcell_pos[lat_ind[2], :][0]
        else:
            column_cor = lat_ind[0] * basis[0][1] + lat_ind[1] * basis[1][1] + unitcell_pos[lat_ind[2], :][1]
        sublattice_ind = int((2 * column_cor) % 3)
        if sublattice_ind == 1:
            psi.apply_local_op(i, "rot_once")
        elif sublattice_ind == 2:
            psi.apply_local_op(i, "rot_twice")

    psi.canonical_form()

    if debug and Lx == 4 and Ly == 4:
        magz = psi.expectation_value("Sz")
        magz_expected = 0.5 * np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1])
        assert ((magz - magz_expected) == 0.0).all()
    if plot:
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        spin_corr_x = CalculateSpinSpinCorrelations(psi)
        ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat)
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        plot_structure_factor(ks, spin_corr_k, lat, ax_corr)
        YC_lat = BuildTriangularLattice(1, 1, SpinHalfSite(None), "finite",
                                        ("open", "open"), "YC")
        YC_lat.plot_brillouin_zone(ax_corr)
        if local:
            plt.show()

    return psi


def GenerateStripeOrderedState(lat, plot=False):
    basis = lat.basis
    aligned_with_x = (basis[0][1] == 0.0 or basis[1][1] == 0.0)
    aligned_with_y = (basis[0][0] == 0.0 or basis[1][0] == 0.0)
    assert(aligned_with_x or aligned_with_y)

    product_state = []
    for i in range(lat.N_sites):
        lat_ind = lat.mps2lat_idx(i)
        if aligned_with_x:
            column_ind = lat_ind[1]
        else:
            column_ind = lat_ind[0]
        product_state.append("up" if column_ind % 2 == 1 else "down")

    psi = MPS.from_product_state(
        lat.mps_sites(),
        product_state,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )
    psi.canonical_form()

    if plot:
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        if local:
            plt.show()

    return psi


def RunDMRG(model, psi_init, dmrg_params=default_dmrg_params,
            plot_convergence=True, print_final_results=True,
            expected_energy=None, results_dir="", energies_fig_title=None):
    E_initial = model.H_MPO.expectation_value(psi_init)
    print("initial energy before dmrg: ", E_initial)
    info = dmrg.run(psi_init, model, dmrg_params)
    E_final = info['E']
    stats = info['sweep_statistics']
    energies = stats['E']
    sweeps = stats['sweep']
    if print_final_results:
        print(f'E = {E_final:.13f}')
        print('final bond dimensions: ', psi_init.chi)
        np.savetxt("Energies.txt", np.array([E_initial] + energies))

    if plot_convergence:
        fig,ax = plt.subplots(figsize=(6, 5))
        ax.plot([-1] + sweeps, [E_initial] + energies, "o")
        ax.set_title("DMRG Sweeps Energies")
        ax.set_xlabel("sweep")
        ax.set_ylabel("E")
        if expected_energy is not None:
            ax.axhline(expected_energy, color="red", linewidth=0.6, alpha=0.5, linestyle="dashed")
        if energies_fig_title is not None:
            fig.savefig(results_dir + energies_fig_title, bbox_inches='tight')

        if local:
            plt.show()
    return E_initial, E_final, sweeps, energies, info


class TriangularXC(tenpy.models.lattice.Lattice):
    dim = 2
    Lu = 2

    def __init__(self, Lx, Ly, sites, spinfull_fermions, **kwargs):

        expected_number = 4 if spinfull_fermions else 2

        try:
            iter(sites)
        except TypeError:
            sites = [sites] * expected_number
        if len(sites) != expected_number:
            raise ValueError(
                "need to specify a single site or exactly {0:d}, got {1:d}".format(expected_number, len(sites)))

        basis = np.array(([1.0, 0.], [0., np.sqrt(3)]))
        delta = np.array([0.5, np.sqrt(3)/2.])

        if spinfull_fermions:
            pos = (-delta / 2 - 0.1 * delta, -delta / 2 + 0.1 * delta, delta / 2 - 0.1 * delta, delta / 2 + 0.1 * delta)
        else:
            pos = (-delta / 2., delta / 2)

        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = []
        nNN = []
        for i in range(len(pos)//2):
            site1 = i
            site2 = i + len(pos)//2
            NN += [(site1, site2, np.array([0, 0])), (site2, site1, np.array([0, 1])), (site1, site1, np.array([1, 0])),
                  (site2, site2, np.array([1, 0])), (site2, site1, np.array([1, 0])), (site2, site1, np.array([1, 1]))]
            nNN += [(site1, site1, np.array([0, 1])), (site2, site2, np.array([0, 1])), (site1, site2, np.array([1, -1])),
                   (site2, site1, np.array([2, 0])), (site1, site2, np.array([1, 0])), (site2, site1, np.array([2, 1]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs.setdefault('bc', ("open", "periodic")) #cylinder
        tenpy.models.lattice.Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


def BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc = ('periodic', 'periodic'), geometry="YC",
                           spinfull_fermions=False):
    if geometry == "YC":
        basis = [[1.0, 0.0], [0.5, sqrt(3) / 2.]]

        unit_cell = [[0.0, 0.0]]
        if spinfull_fermions:
            unit_cell = [[-0.1, 0.0], [0.1, 0.0]]

        nearest_neighbors = []
        next_nearest_neighbors = []
        num_flavors = 2 if spinfull_fermions else 1
        for i in range(num_flavors):
            nearest_neighbors += [[i, i, [1, 0]], [i, i, [0, 1]],
                                 [i, i, [-1, 1]]]
            next_nearest_neighbors += [[i, i, [1, 1]], [i, i, [-1, 2]],
                                       [i, i, [-2, 1]]]

        triangular_lat = tenpy.models.lattice.Lattice([Lx, Ly], [site]*len(unit_cell), basis=basis,
                                        positions=unit_cell, bc=bc, pairs={'nearest_neighbors': nearest_neighbors,
                                                                           'next_nearest_neighbors': next_nearest_neighbors},
                                        bc_MPS=bc_MPS)

        return triangular_lat
    elif geometry == "XC":
        triangular_lat = TriangularXC(Lx, Ly, site, spinfull_fermions, bc_MPS=bc_MPS, bc=bc)
        return triangular_lat
    else:
        raise ValueError("unrecognized geometry")

def LxInfiniteMPSCorrelations(Lx, Ly):
    return Lx * max(1, Ly//Lx) # integer multiple of the unitcell, taking Lx_large, Ly to be close

def BuildSpinTriangularLatticeWrap(Lx, Ly, bc_MPS, conserve, bc, geometry):
    Lx_correlations = Lx
    if bc_MPS == "infinite":
        Lx_correlations = LxInfiniteMPSCorrelations(Lx, Ly)
    site = SpinHalfSite(conserve=conserve)
    return BuildTriangularLattice(Lx_correlations, Ly, site, bc_MPS, bc=bc, geometry=geometry)


def initialStateFromFile(initial_state):
    return "from_file" in initial_state


def GetTriangularLatticeInitialState(initial_state, triangular_lat, initial_psi_dir, abs_magz):
    Lx = triangular_lat.Ls[0]
    Ly = triangular_lat.Ls[1]
    N_sites = triangular_lat.N_sites
    assert(abs_magz <= N_sites // 2), "normalized magnetization cannot exceed 1"
    if abs_magz > 0:
        assert(initial_state == "Random" or initialStateFromFile(initial_state))
    if initial_state == "Random":
        N_up = N_sites // 2 + abs_magz
        N_down = N_sites - N_up
        down_indices = np.random.choice(np.arange(0, N_sites), N_down, replace=False)
        print(down_indices)
        down_indices = np.array(down_indices)
        product_state = ["up"] * N_sites
        for down_ind in down_indices:
            product_state[down_ind] = "down"
        psi = MPS.from_product_state(
            triangular_lat.mps_sites(),
            product_state,
            bc=triangular_lat.bc_MPS,
            unit_cell_width=triangular_lat.mps_unit_cell_width,
        )

    elif initial_state == "120":
        psi = Generate120DegOrderedState(lat=triangular_lat, Lx=Lx, Ly=Ly)

    elif initial_state == "stripe":
        psi = GenerateStripeOrderedState(lat=triangular_lat)

    elif initialStateFromFile(initial_state):
        initial_psi_path = initial_psi_dir + 'psi_gs.pkl'
        print(f"loading initial state: {initial_psi_path}")
        with open(initial_psi_path, 'rb') as psi_load:
            psi = pickle.load(psi_load)
        #psi.canonical_form()

    else:
        raise ValueError("unrecognized initial state")
        psi = None
    return psi


def TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, initial_state, conserve, J2, geometry, chi, max_sweeps, norm_magz,
                              Delz=1.0):
    bc_string = ""
    for bc_ax in bc:
        if bc_ax == "periodic":
            bc_string += "p"
        else:
            assert(bc_ax == "open")
            bc_string += "o"

    geometry_dir = f"Lx_{Lx}_Ly_{Ly}_bc_{bc_string}_{geometry}/"
    params_dir = f"{bc_MPS}_init_{initial_state}_conserve_{conserve}_J2_{J2}"
    if Delz is not None and abs(float(Delz) - 1.0) > 1e-15:
        params_dir += f"_Delz_{Delz}"
    if chi is not None:
        params_dir += f"_chi_{chi}"
    if max_sweeps is not None:
        params_dir += f"_maxsweeps_{max_sweeps}"
    if float(norm_magz) > 1e-15:
        norm_magz_float = float(norm_magz)
        params_dir += f"_magz_{norm_magz_float:.4f}"
    return geometry_dir, params_dir + "/"


def CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc, bc_MPS, initial_state, conserve, J2, geometry,
                            chi_max=None, max_sweeps=None, norm_magz=0.0, Delz=1.0):
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    geometry_dir, params_dir = TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, initial_state, conserve, J2, geometry,
                                                         chi_max, max_sweeps, norm_magz, Delz)
    results_dir = main_results_dir + geometry_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_dir += params_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir


def GenerateJ1J2SpinTriangularModel(J2, Delz, triangular_lat):
    J1 = 1.0
    assert(len(triangular_lat.pairs["nearest_neighbors"]) > 0 and
           len(triangular_lat.pairs["next_nearest_neighbors"]) > 0)

    J1_xy = J1
    J1_z = J1 * Delz
    J2_xy = J2
    J2_z = J2 * Delz

    J1J2_model = SpinModel({"lattice": triangular_lat, "Jx": J1_xy, "Jy": J1_xy, "Jz": J1_z})
    J1J2_model.manually_call_init_H = True

    nnn_couplings_list = []
    if abs(J2) > 0.0:
        for u1, u2, dx in triangular_lat.pairs['next_nearest_neighbors']:
            AddAndTrackCoupling(J1J2_model, 0.5 * J2_xy, u1, "Sp", u2, "Sm", dx,
                                nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, 0.5 * J2_xy, u1, "Sm", u2, "Sp", dx,
                                nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, J2_z, u1, "Sz", u2, "Sz", dx,
                                nnn_couplings_list)
    J1J2_model.init_H_from_terms()
    return J1J2_model, nnn_couplings_list


def calculateGutzwillerEnergyTriangularJ1J2(gutz_results_dir, Lx, Ly, chi, flux, bc_MPS, J2, Delz, bc, geometry,
                                            gs_manifold_index, norm_magz, monopole_Q, reorder_lattice=False, model_type=None):
    psi_path = CreateGutzwillerCaseDir(gutz_results_dir, Lx, Ly, chi, flux, geometry,
                                       bc_MPS, gs_manifold_index, model_type, norm_magz, monopole_Q) + "/psi_gutzwiller.pkl"

    print(f"calculating energy for MPS in path {psi_path} with triangular J1J2 model for J2={J2}")
    site = SpinHalfSite(conserve="Sz")

    with open(psi_path, 'rb') as f:
        psi = pickle.load(f)
    
    finite = (bc_MPS == "finite")
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc, geometry=geometry)
    J1J2_model, _ = GenerateJ1J2SpinTriangularModel(J2, Delz, triangular_lat)
    if reorder_lattice:
        exit("reorder lattice not supported anymore")

    print(triangular_lat.N_sites)
    E = J1J2_model.H_MPO.expectation_value(psi)
    if bc_MPS == "finite":
        E /= triangular_lat.N_sites
    print("Energy: ", E)
    return E


def SaveSimulationOutput(results_dir, spin_corr_x, ks, spin_corr_k, fig_corr_k, fig_lat,
                         special_points_structure_factor=None, lattice=None):
    np.savetxt(results_dir + "spin_corr_x.csv", spin_corr_x)
    np.savetxt(results_dir + "spin_corr_k.csv", spin_corr_k)
    np.savetxt(results_dir + "ks.csv", ks)
    fig_corr_k.savefig(results_dir + "momentum_space_correlations.png", bbox_inches='tight')
    fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')
    if special_points_structure_factor is not None:
        np.savetxt(results_dir + "special_points_structure_factor.csv", special_points_structure_factor)

    if lattice is not None:
        with open(results_dir + 'lattice.pkl', 'wb') as f_lat:
            pickle.dump(lattice, f_lat)


def calculateStructureFactorAtSpecialPoints(lat, spin_corr_x):
    special_bz_points = getSpecielBzPoints()
    special_points_structure_factor = np.zeros((len(special_bz_points.keys()), 3))
    for ind, key in enumerate(special_bz_points.keys()):
        k = special_bz_points[key]
        sf_at_special_point = structure_factor(spin_corr_x, lat, k)
        special_points_structure_factor[ind, 0:2] = k
        special_points_structure_factor[ind, 2] = sf_at_special_point
    return special_points_structure_factor


def TriangularJ1J2DMRG(Lx, Ly, bc, bc_MPS, conserve=True, initial_state="Random", J2=0.0, geometry="YC",
                       chi_max=None, initial_psi_dir=None, max_sweeps=None, norm_magz=0.0, Delz=1.0):
    if isinstance(bc, str):
        bc_parsed = bc.split("-")
        bc = (bc_parsed[0], bc_parsed[1])

    if local:
        main_results_dir = glob_results_dir + "LocalJ1J2TriangularDMRGResults/"
        results_dir = CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc, bc_MPS, initial_state, conserve, J2,
                                              geometry, chi_max, max_sweeps, norm_magz, Delz)
    else:
        results_dir = "./"

    if conserve:
        site = SpinHalfSite(conserve='Sz')
    else:
        site = SpinHalfSite(conserve=None)

    triangular_lat = BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc=bc, geometry=geometry)
    abs_magz = AbsMagzFromNormMagz(norm_magz, triangular_lat.N_sites)
    center_site_mps_index = triangular_lat.lat2mps_idx([Lx // 2, Ly // 2, 0])
    print("center site mps index: ", center_site_mps_index)

    J1J2_model, nnn_couplings_list = GenerateJ1J2SpinTriangularModel(J2, Delz, triangular_lat)

    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    PlotLattice(triangular_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)

    print_couplings = False
    if print_couplings:
        PrintCouplings(J1J2_model)
        if local:
            plt.show()
        exit(1)

    if local:
            plt.show()

    psi = GetTriangularLatticeInitialState(initial_state, triangular_lat, initial_psi_dir, abs_magz)
   
    dmrg_params = default_dmrg_params

    if chi_max is not None:
        ChangeChiInDMRGParams(dmrg_params, chi_max)
    chi_max = dmrg_params['trunc_params']['chi_max']

    if max_sweeps is not None:
        dmrg_params['max_sweeps'] = max_sweeps

    if initialStateFromFile(initial_state):
        chi_max_psi = np.max(psi.chi)
        chi_max = int(max(chi_max_psi, chi_max))

        ChangeChiInDMRGParams(dmrg_params, chi_max)
        dmrg_params['chi_list'] = {0:chi_max}
        dmrg_params['min_sweeps'] = 3

    with open(results_dir + "dmrg_params.json", "w") as f:
        json.dump(dmrg_params, f, indent=4)

    if initialStateFromFile(initial_state):
        with open("psi_initial_dir.txt", 'w') as f:
            f.write(initial_psi_dir)
    else:
        with open(results_dir + 'psi_initial' + ".pkl", 'wb') as f:
            pickle.dump(psi, f)

    RunDMRG(J1J2_model, psi, dmrg_params=dmrg_params, print_final_results=True, results_dir=results_dir,
            energies_fig_title="energies.png")

    psi.canonical_form()
    E_final = J1J2_model.H_MPO.expectation_value(psi)
    print(f"Energy calculated from full MPO: {E_final}")
    with open(results_dir + 'psi_gs' + ".pkl", 'wb') as f:
        pickle.dump(psi, f)

    sites1, sites2 = None, None
    lat_for_corr = triangular_lat
    if bc_MPS == "infinite":
        Lx_large = LxInfiniteMPSCorrelations(Lx, Ly)
        sites1 = np.arange(0, Ly * Lx_large)
        sites2 = np.arange(0, Ly * Lx_large)
        lat_for_corr = BuildTriangularLattice(Lx_large, Ly, site, bc_MPS, bc=bc, geometry=geometry)

    spin_corr_x = CalculateSpinSpinCorrelations(psi, sites1, sites2)
    ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat_for_corr, assert_realness=False)

    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    plot_structure_factor(ks, spin_corr_k, triangular_lat, ax_corr)

    YC_lat = BuildTriangularLattice(1, 1, SpinHalfSite(None), "finite", ("open", "open"), "YC")
    YC_lat.plot_brillouin_zone(ax_corr)

    special_points_structure_factor = calculateStructureFactorAtSpecialPoints(triangular_lat, spin_corr_x)

    SaveSimulationOutput(results_dir, spin_corr_x, ks, spin_corr_k, fig_corr, fig_lat,
                         special_points_structure_factor=special_points_structure_factor, lattice=lat_for_corr)


def ExplicitMPSNorm(mps):
    tensors = mps._B if (mps.bc == "finite") else [mps.get_B(i) for i in range(mps.L)]
    first_tensor = tensors[0].to_ndarray()
    norm_transfer = np.eye(first_tensor.shape[0], dtype=np.result_type(first_tensor.dtype, np.complex128))
    for B in tensors:
        B_arr = B.to_ndarray()
        ket_contracted = np.tensordot(norm_transfer, B_arr, axes=([1], [0]))
        norm_transfer = np.tensordot(B_arr.conj(), ket_contracted, axes=([0, 1], [0, 1]))

    if mps.bc == "finite":
        assert norm_transfer.shape == (1, 1)
        return norm_transfer[0, 0]
    return norm_transfer


def calculateOverlapBetweenGutzwillerAndDMRG(dmrg_dir, gutzwiller_dir,
                                             psi_gutz_fname='psi_gutzwiller.pkl'):

    with open(dmrg_dir + 'psi_gs.pkl', 'rb') as f_dmrg:
        psi_dmrg = pickle.load(f_dmrg)
    with open(gutzwiller_dir + psi_gutz_fname, 'rb') as f_gutz:
        psi_gutz = pickle.load(f_gutz)
    
    #compressed_chi = 2000
    #max_trunc_err_dmrg = psi_dmrg.compress({"compression_method":'SVD', "trunc_params":{"chi_min":compressed_chi, "chi_max":compressed_chi}})
    #max_trunc_err_gutz = psi_gutz.compress({"compression_method":'SVD', "trunc_params":{"chi_min":compressed_chi, "chi_max":compressed_chi}})
    #print(f"compressed dmrg psi with max truncation error {max_trunc_err_dmrg} and gutzwiller psi with max truncation error {max_trunc_err_gutz}")

    #overlap = abs(psi_dmrg.overlap(psi_gutz, num_ev=4))
    print(f"calculating overlap between dmrg wavefunction in {dmrg_dir} and gutzwiller wavefunction in {gutzwiller_dir}")
    overlap = psi_dmrg.overlap(psi_gutz)
    print(f"overlap is {overlap}, |overlap| is {abs(overlap)}")
    return overlap


def ComputeCorrelationsFromMPSFile(psi_dir, Lx, Ly, bc, geometry="YC", psi_fname="psi_gs.pkl",
                                   transverse_correlations=False, plot_mode="interpolate"):
    with open(psi_dir + psi_fname, 'rb') as f:
        psi = pickle.load(f)

    bc_MPS = "finite" if psi.finite else "infinite"
    conserve = psi.sites[0].conserve
    triangular_lattice = BuildSpinTriangularLatticeWrap(Lx, Ly, bc_MPS, conserve, bc, geometry)

    spin_corr_x = CalculateSpinSpinCorrelations(psi, sites1=np.arange(0, triangular_lattice.N_sites),
                                                sites2=np.arange(0, triangular_lattice.N_sites),
                                                transverse_correlations=transverse_correlations)
    ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lattice,
                                                          assert_realness=True)

    # spin_corr_k_from_file = np.loadtxt(psi_dir + "spin_corr_k.csv", dtype=complex)
    # print("largest diff: ", np.max(np.abs(spin_corr_k - spin_corr_k_from_file)))
    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"Spin Structure Factor"
    ax.set_title(title)
    plot_structure_factor(ks, spin_corr_k, triangular_lattice, ax, mode=plot_mode)
    triangular_lattice.plot_brillouin_zone(ax)
    fig.savefig(psi_dir + "momentum_space_correlations_local", bbox_inches='tight')
    if local:
            plt.show()


def PlotCorrelationsFromFiles(results_dir,
                              energy_ax=None, initial_state="", show_energies=True, output_dir=None,
                              fig_title="", k_space=True, plot_mode="interpolate"):
    if show_energies:
        if energy_ax is None:
            fig, energy_ax = plt.subplots()
        energies = np.loadtxt(results_dir + "Energies.txt")
        energy_ax.plot(energies, "o", label=initial_state)

    fig, ax = plt.subplots(figsize=(6, 5))
    if k_space:
        new_format = os.path.isfile(results_dir + "ks.csv")
        if new_format:
            ks = np.loadtxt(results_dir + "ks.csv")
            corr_k = np.loadtxt(results_dir + "spin_corr_k.csv")
            with open(results_dir + "lattice.pkl", 'rb') as f:
                lattice = pickle.load(f)
            plot_structure_factor(ks, corr_k, lattice, ax, mode=plot_mode)
        else:
            corr_k = np.loadtxt(results_dir + "spin_corr_k.csv", dtype=np.complex128)
            Kx = np.loadtxt(results_dir + "Kx.csv")
            Ky = np.loadtxt(results_dir + "Ky.csv")
            triangular_lat = BuildTriangularLattice(2, 2, SpinHalfSite(conserve='Sz'), "finite")
            triangular_lat.plot_brillouin_zone(ax)
            assert (np.max(np.abs(np.imag(corr_k))) < 1e-14)
            ImshowMatrix(ax, fig, Kx, Ky, np.real(corr_k), title="Spin Correlations")
        YC_lat = BuildTriangularLattice(1, 1, SpinHalfSite(None), "finite", ("open", "open"), "YC")
        YC_lat.plot_brillouin_zone(ax)
    else:
        corr_x = np.loadtxt(results_dir + "spin_corr_x.csv", dtype=np.complex128)
        ImshowMatrix(ax, fig, np.array([0,1]), np.array([0,1]), corr_x, xlabel="x/L",
                     ylabel="y/L")

    if output_dir is None:
        fig_fname = "momentum_space_correlations_local.png" if k_space else "momentum_space_correlations_local.png"
        fig.savefig(results_dir + fig_fname, bbox_inches='tight')
    else:
        fig_fname = output_dir + f"correlations_{fig_title}.png"
        fig.savefig(fig_fname, bbox_inches='tight')



def PlotRealSpaceCorrelations(results_dir):
    corr_x = np.loadtxt(results_dir + "spin_corr_x.csv", dtype=np.complex128)
    print(corr_x.shape)
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])
    fig, ax = plt.subplots(figsize=(6, 5))
    ImshowMatrix(ax, fig, X, Y, np.real(corr_x), label="real part")
    ax.legend()


def _parameter_plot_label(parameter_name):
    parameter_labels = {
        "J2": r"$J_2$",
        "Delz": r"$\Delta_z$",
    }
    return parameter_labels.get(parameter_name, parameter_name)


def GutzwillerDMRGOverlaps(scanned_parameter_name, scanned_parameter_values, gutz_parent_dir, Lx, Ly, gutz_chi_max,
                           gutz_flux, gutz_mon_Q, output_dir, dmrg_initial_state, dmrg_parent_dir, geometry, bc_MPS,
                           gutz_gs_manifold_index, dmrg_chi_max, dmrg_max_sweeps, dmrg_conserve, model_type, norm_magz,
                           **kwargs):
    overlaps = []
    dmrg_energies = []
    gutz_energies = []
    gutz_case_dir = CreateGutzwillerCaseDir(gutz_parent_dir, Lx, Ly, gutz_chi_max, gutz_flux, geometry,
                                            bc_MPS, gutz_gs_manifold_index, model_type=model_type, norm_magz=norm_magz,
                                            monopole_Q=gutz_mon_Q)
    finite = (bc_MPS == "finite")
    bc = ("open", "periodic") if finite else ("periodic", "periodic")
    
    with open(output_dir + "parent_directories.txt", 'w') as f:
        f.write(f"dmrg parent dir: {dmrg_parent_dir}\n")
        f.write(f"gutzwiller parent dir: {gutz_parent_dir}\n")

    for parameter_value in scanned_parameter_values:
        hamiltonian_params = dict(kwargs)
        hamiltonian_params[scanned_parameter_name] = parameter_value
        J2 = hamiltonian_params.get("J2")
        Delz = hamiltonian_params.get("Delz", 1.0)
        if J2 is None:
            raise ValueError("J2 must be supplied directly or scanned through the parameter file")
        if Delz is None:
            raise ValueError("Delz must be supplied directly or scanned through the parameter file")

        dmrg_geom_dir, dmrg_params_dir = (
            TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, dmrg_initial_state, dmrg_conserve, J2, geometry,
                                      dmrg_chi_max, dmrg_max_sweeps, norm_magz, Delz))
        dmrg_dir = dmrg_parent_dir + dmrg_geom_dir + dmrg_params_dir
        
        unitcell_width = 2 if geometry == "XC" else 1
        
        sweep_energies = np.loadtxt(dmrg_dir + "Energies.txt", dtype=np.float64)
        dmrg_energy = sweep_energies[-1]
        if finite:
            dmrg_energy /= (Lx * Ly * unitcell_width)
        dmrg_energies.append(dmrg_energy)

        gutz_energy = calculateGutzwillerEnergyTriangularJ1J2(gutz_parent_dir, Lx, Ly, gutz_chi_max, gutz_flux, bc_MPS,
                                                              J2, Delz, bc, geometry, gutz_gs_manifold_index, norm_magz,
                                                              gutz_mon_Q, model_type=model_type)
        gutz_energies.append(gutz_energy)

        PlotCorrelationsFromFiles(dmrg_dir, show_energies=False, output_dir=output_dir,
                                  fig_title=f"dmrg_{scanned_parameter_name}_{parameter_value}")

        overlap_J2 = calculateOverlapBetweenGutzwillerAndDMRG(dmrg_dir, gutz_case_dir)
        overlaps.append(overlap_J2)


    scanned_parameter_values = np.array(scanned_parameter_values)
    overlaps = np.array(overlaps)
    dmrg_energies = np.array(dmrg_energies)
    gutz_energies = np.array(gutz_energies)
    data = np.column_stack((scanned_parameter_values, overlaps, dmrg_energies, gutz_energies))
    np.savetxt(output_dir + "data.txt", data, header=f'{scanned_parameter_name} overlap E_DMRG E_Gutzwiller')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(scanned_parameter_values, overlaps, "o")
    ax.set_xlabel(_parameter_plot_label(scanned_parameter_name))
    ax.set_ylabel("overlap")
    fig.savefig(output_dir + f"overlaps_initial_state_{dmrg_initial_state}.png", bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(scanned_parameter_values, dmrg_energies, "ro", label="dmrg")
    ax.plot(scanned_parameter_values, gutz_energies, "bo", label="Gutzwiller")
    ax.set_xlabel(_parameter_plot_label(scanned_parameter_name))
    ax.set_ylabel(r"$E$")
    ax.legend()
    fig.savefig(output_dir + f"energies_initial_state_{dmrg_initial_state}.png", bbox_inches='tight')


def linear_model(x, m, b):
    return m * x + b


def FitLinearModel(x, y):
    popt, pcov = curve_fit(linear_model, x, y)
    m = popt[0]
    b = popt[1]
    x0 = (-1) * b / m
    dm = np.sqrt(pcov[0, 0])
    db = np.sqrt(pcov[1, 1])
    dx0 = np.sqrt((db / m) ** 2 + (dm * b / (m ** 2)) ** 2)

    return {"m": m, "b": b, "db": db, "dm": dm}


def GutzwillerBondDimensionScaling(gutz_results_dir, Lx, Ly, chis, flux,
                                   output_dir):
    Es = []
    inv_chis = np.array(1. / np.array(chis))
    bc_MPS = "finite"
    J2 = 0.125
    bc = ("open", "periodic")
    geometry = "YC"
    gs_manifold_index = 0
    Delz = 1.0
    for chi in chis:
        E = calculateGutzwillerEnergyTriangularJ1J2(gutz_results_dir, Lx, Ly, chi, flux, bc_MPS, J2, Delz, bc, geometry,
                                                    gs_manifold_index, None, None)
        Es.append(E)

    fit_params = FitLinearModel(inv_chis, Es)
    m, b, db, dm = fit_params["m"], fit_params["b"], fit_params["db"], fit_params["dm"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(inv_chis, Es, "bo")

    inv_chis_plot_fit = np.linspace(0.0, np.max(inv_chis), 2)
    ax.plot(inv_chis_plot_fit, m*inv_chis_plot_fit + b, "b--")
    ax.errorbar([0.0], [b], yerr=[db], fmt="ro")

    ax.set_xlabel(r"$1 / \chi$")
    ax.set_ylabel(r"$E$")
    print(f"Energy for chi=inf: {b} +- {db}")
    fig.savefig(output_dir + "Gutzwiller_E_chi_scaling.png", bbox_inches="tight")
    if local:
            plt.show()


def DMRGCorrelations():
    main_results_dir = "NewTenpyTriangularLatticeResults/"
    case_dir = f"Lx_6_Ly_6_bc_op/finite_init_Random_conserve_1_J2_{0.0}/"
    PlotCorrelationsFromFiles(glob_results_dir + main_results_dir + case_dir)
    if local:
            plt.show()

    for J2 in [0.1, 0.11, 0.12, 0.125, 0.13, 0.14, 0.15]:
        fig, energy_ax = plt.subplots(figsize=(6, 5))
        for initial_state in ["stripe", "Random"]:
            case_dir = f"Lx_6_Ly_6_bc_op/finite_0.0_init_{initial_state}_conserve_1_J2_{J2}/"
            PlotCorrelationsFromFiles(glob_results_dir + main_results_dir + case_dir, energy_ax=energy_ax,
                                      initial_state=initial_state)
        energy_ax.legend()
        if local:
            plt.show()



def TryCylinderFlux():
    lat = BuildTriangularLattice(2, 3, FermionSite(), "finite")
    fermion_model = CouplingModel(lat)
    for u1, u2, dx in lat.pairs['nearest_neighbors']:
        strength_with_flux = fermion_model.coupling_strength_add_ext_flux(1.0, dx, [0, pi / 4])
        fermion_model.add_coupling(strength_with_flux, u1, 'Cd', u2, 'C', dx)
        fermion_model.add_coupling(np.conj(strength_with_flux), u2, 'Cd', u1, 'C', -np.array(dx))
    fig, ax = plt.subplots(figsize=(6, 5))
    PlotLattice(lat, ax)
    if local:
            plt.show()


def CompareGutzwillerGroundStateSectorsXC():
    gutz_dir = glob_results_dir + "LocalGutzwillerResults/Dirac_finite_Lx_40_Ly_2_chi_1000_flux_0.0_XC_gsindex_"
    for i in range(4):
        i_dir = gutz_dir + f"{i}/"
        PlotCorrelationsFromFiles(i_dir, show_energies=False, output_dir=i_dir)
        for j in range(i + 1, 4):
            psi1_path = i_dir + "psi_gutzwiller.pkl"
            psi2_path = gutz_dir + f"{j}/" + "psi_gutzwiller.pkl"
            with open(psi1_path, 'rb') as psi_load:
                psi1 = pickle.load(psi_load)
            with open(psi2_path, 'rb') as psi_load:
                psi2 = pickle.load(psi_load)
            print(f"overlap {i},{j}: ", psi1.overlap(psi2))

            C_i = np.loadtxt(gutz_dir + f"{i}/C_slater.csv")
            C_j = np.loadtxt(gutz_dir + f"{j}/C_slater.csv")
            fig, ax = plt.subplots(figsize=(6, 5))
            ImshowMatrix(ax, fig, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.abs(C_i - C_j),
                         xlabel="x/L", ylabel="y/L")
            fig.savefig(f"C_diff_{i}{j}.png", bbox_inches='tight')
            plt.show()



def calculateZ2EntanglementEntropy():
    parent_dir = glob_results_dir + "Z2_Topological_EE/"
    cases = []
    central_bonds = []
    chis = [2000, 3000, 4000, 6000]
    Lx = 2
    Lys = [4, 5, 6, 7]
    for i_Ly, Ly in enumerate(Lys):
        cases.append(parent_dir + f"Z2_infinite_Lx_{Lx}_Ly_{Ly}_chi_{chis[i_Ly]}_flux_0.0_YC_gsindex_0/")
        central_bonds.append(Lx * Ly // 2)

    EEs = []
    for i_case, case in enumerate(cases):
        with open(case + "psi_gutzwiller.pkl", 'rb') as f:
            psi = pickle.load(f)
        EE_central_bond = psi.entanglement_entropy(bonds=[central_bonds[i_case]])[0]
        EEs.append(EE_central_bond)

    Lys = np.array(Lys)
    EEs = np.array(EEs)
    plt.plot(Lys, EEs, "o")

    fit_params = FitLinearModel(Lys, EEs)
    m, b, db, dm = fit_params["m"], fit_params["b"], fit_params["db"], fit_params["dm"]
    print(f"Constant EE contribution: {b} +- {db}")
    Ly_linplot = np.array([0, np.max(Lys)])
    plt.plot(Ly_linplot, m*Ly_linplot + b, "-")

    plt.show()


def getEnergyDifferenceBetweenSectors(dir1, dir2, title, dmrg, fig_name):
    data1 = np.loadtxt(dir1 + "data.txt")
    data2 = np.loadtxt(dir2 + "data.txt")
    J2 = data1[:, 0]
    assert(np.max(np.abs(J2 - data2[:,0])) <= 0.0)
    E_dmrg_1 = data1[:,2]
    E_dmrg_2 = data2[:,2]
    E_gutz_1 = data1[:, 3]
    E_gutz_2 = data2[:, 3]
    fig, ax = plt.subplots(figsize=(6, 5))
    if dmrg:
        ax.plot(J2, E_dmrg_1 - E_dmrg_2, "ro", label="DMRG")
    else:
        ax.plot(J2, E_gutz_1 - E_gutz_2, "bo", label="Gutzwiller")
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$\Delta$E")
    ax.set_title(title)
    fig.savefig(fig_name, bbox_inches='tight')
    plt.show()


def calculateMonopoleEnergies(parent_dir, norm_magz, mon_Qs):
    assert(mon_Qs[0] == 0)
    Es = []
    fig, ax = plt.subplots(figsize=(6,5))
    Lx, Ly = 6, 6
    flux = 0.0
    chi = 2000
    J2 = 0.125
    Delz = 1.0
    for monopole_Q in mon_Qs:
        E = calculateGutzwillerEnergyTriangularJ1J2(parent_dir, Lx, Ly, chi, flux,
                                                    "finite", J2, Delz, ("open", "periodic"),
                                                    "YC", 0, norm_magz,
                                                    monopole_Q, model_type=model_type_dirac,
                                                    norm_magz=norm_magz, monopole_Q=monopole_Q)
        Es.append(E)

    Es = np.array(Es)
    mon_Qs = np.array(mon_Qs)
    E_fermi_pocket = Es[0]
    e_diff = (Es[1:] - E_fermi_pocket) / abs(E_fermi_pocket)
    ax.plot(mon_Qs[1:], e_diff, "o")
    ax.set_xlabel(r"$Q[2\pi/N]$")
    ax.set_ylabel(r"$\delta E[J_1] / E_{fp}$")
    ax.set_title("Energy vs. Monopole Flux for J2/J1=1/8")
    return e_diff, ax, fig


def calculateOverlapsFastLocal(Lx, Ly, Q, magz):
    dir_gutz = monopole_dir + f"Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz}_monQ_{Q}/"
    dir_dmrg = glob_results_dir + (f"TriangularJ1J2DMRG_6_15_e476229/Lx_{Lx}_Ly_{Ly}_bc_op_YC/"
                       f"finite_init_Random_conserve_1_J2_0.1_chi_3500_maxsweeps_50_magz_{magz}/")

    with open(dir_gutz + "psi_gutzwiller.pkl", 'rb') as f:
        psi_gutz = pickle.load(f)
    with open(dir_dmrg + "psi_gs.pkl", 'rb') as f:
        psi_dmrg = pickle.load(f)

    # psi_gutz.unit_cell_width = psi_dmrg.unit_cell_width

    print(f"overlap with magz={magz}, Q={Q}: {np.abs(psi_dmrg.overlap(psi_gutz))}")


def plotMonopoleOrderParameter():
    Lx, Ly = 6,6

    lattice = BuildTriangularLattice(Lx, Ly, SpinHalfSite("Sz"), "finite", bc=("open", "periodic"))

    magz1 = 0.056
    Q1 = 0
    magz2 = 0.111
    Q2 = 0

    dir1 = monopole_dir + f"Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz1}_monQ_{Q1}/"
    dir2 = monopole_dir + f"Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz2}_monQ_{Q2}/"

    with open(dir1 + "psi_gutzwiller.pkl", 'rb') as f:
        psi1 = pickle.load(f)
    with open(dir2 + "psi_gutzwiller.pkl", 'rb') as f:
        psi2 = pickle.load(f)

    psi1.unit_cell_width = lattice.mps_unit_cell_width
    psi2.unit_cell_width = lattice.mps_unit_cell_width
    env = MPSEnvironment(bra=psi2, ket=psi1)
    local_vals = env.expectation_value("Sp")

    Kx, Ky, monopole_op_k = ComputeMomentumSpaceStructureFactor(local_vals, lattice, assert_realness=False,
                                                                transform_expectation_value=True,
                                                                new_implementation=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ImshowMatrix(ax, fig, Kx, Ky, np.abs(monopole_op_k), title=r"Fourier Transform of $<n+1|S^+|n>$")
    lattice.plot_brillouin_zone(ax)
    fig.savefig(meetings_dir + f"20_6_2026/monopole_OP/LX_{Lx}_Ly_{Ly}_magzlow_{magz1}_Q1_{Q1}_Q2_{Q2}.png", bbox_inches='tight')
    plt.show()


def getStructureFactorPeaksFromPostProcessDir(results_file, n_sites_cases, dict):
    sf_data = np.loadtxt(results_file, dtype=np.float64)
    for j in range(sf_data.shape[0]):
        case_index = int(sf_data[j, 0])
        n_sites = n_sites_cases[case_index]
        k_2D = np.array([sf_data[j, 2], sf_data[j, 3]])
        k_name = getKPointName(k_2D)
        sf = sf_data[j, 4]
        dict[k_name][case_index, :] = [n_sites, sf]


def getStructureFactorPeaksFromRunDir(results_file, n_sites_cases, dict, Ly, monQs=None):
    assert(Ly > 0)
    Lx_cases = n_sites_cases / Ly
    for ind, n_sites in enumerate(n_sites_cases):
        _results_file = results_file.replace("Lx_#", f"Lx_{int(Lx_cases[ind])}")
        if monQs is not None:
            _results_file = _results_file.replace("monQ_#", f"monQ_{monQs[ind]}")

        sf_data = np.loadtxt(_results_file, dtype=np.float64)
        for j in range(sf_data.shape[0]):
            k_2D = np.array([sf_data[j, 0], sf_data[j, 1]])
            k_name = getKPointName(k_2D)
            sf = sf_data[j, 2]
            dict[k_name][ind, :] = [n_sites, sf]


def AnalyzeMagnetizedJ1J2Correlations(results_dir, n_sites_cases, postprocess_dir=True, Ly=0, monQs=None):
    # magz1_dir = glob_results_dir + "PostProcess/StructureFactor_Lx12_Ly6_infinite_special_points_structure_factor/"
    #magz1_dir = glob_results_dir + \
    #            "PostProcess/StructureFactorScaling_Gutzwiller_MonQ3_Magz2_special_points_structure_factor/"
    #magz2_dir = glob_results_dir + \
    #            "PostProcess/StructureFactor_Lx12_Ly6_infinite_magz2_special_points_structure_factor/"
    #dirs = [magz1_dir, magz2_dir]

    special_k_points = getSpecielBzPoints()
    fig, ax = plt.subplots(figsize=(6, 5))

    sf_dict = {k_name:np.zeros((n_sites_cases.shape[0], 2)) for k_name in special_k_points.keys()}
    if postprocess_dir:
        sf_filename = "special_points_structure_factor.txt"
        getStructureFactorPeaksFromPostProcessDir(results_dir + sf_filename, n_sites_cases, sf_dict)
    else:
        sf_filename = "special_points_structure_factor.csv"
        getStructureFactorPeaksFromRunDir(results_dir + sf_filename, n_sites_cases, sf_dict, Ly, monQs)


    colors = ["red", "blue", "green", "cyan", "magenta"]
    min_sf = np.inf
    max_sf = -np.inf
    for i_key,key in enumerate(sf_dict.keys()):
        if key == "K_prime":
            continue
        Ns = sf_dict[key][:, 0]
        scaled_sf = sf_dict[key][:,1] / Ns
        min_sf = min(np.min(scaled_sf), min_sf)
        max_sf = max(np.max(scaled_sf), max_sf)
        N_inv_sqrt = 1. / Ns**0.5
        ax.plot(N_inv_sqrt, scaled_sf, "o", color=colors[i_key], label=key)
        lin_fit = FitLinearModel(N_inv_sqrt, scaled_sf)
        N_inv_sqrt_fit_plot = np.array([0.0, np.max(N_inv_sqrt)])
        ax.plot(N_inv_sqrt_fit_plot, linear_model(N_inv_sqrt_fit_plot, lin_fit["m"], lin_fit["b"]), "--",
                color=colors[i_key])


    ax.set_xlabel(r"$1/N^{0.5}$")
    ax.set_ylabel(r"$S_{tran}(Q)$")
    ax.set_xlim((0., 0.2))
    ax.set_ylim((0.0, 1.1*max_sf))

    fig.tight_layout()
    plt.legend()



if __name__ == "__main__":
    magz1_dir = glob_results_dir + \
               "PostProcess/StructureFactorScaling_Gutzwiller_MonQ3_Magz2_special_points_structure_factor/"
    magz2_dir = glob_results_dir + \
               "PostProcess/StructureFactor_Lx12_Ly6_infinite_magz2_special_points_structure_factor/"
    Ly = 6
    ns = Ly * np.array([12, 14, 16, 18])
    #AnalyzeMagnetizedJ1J2Correlations(magz1_dir, ns)
    #AnalyzeMagnetizedJ1J2Correlations(magz2_dir, ns)
    #plt.show()


    #TriangularJ1J2DMRG(8, 3, ("open", "periodic"), "finite", J2=0.0, chi_max=400, max_sweeps=15,
    #                   Delz=2.0)
    norm_magz_third_dir = glob_results_dir + \
               "PiFluxGutzwiller_8_28/Dirac_finite_Lx_#_Ly_6_chi_8000_flux_0.0_YC_gsindex_0_magz_0.3330_monQ_#_svdmin_0.002/"
    AnalyzeMagnetizedJ1J2Correlations(norm_magz_third_dir, ns, postprocess_dir=False, Ly=Ly,
                                      monQs=(ns/Ly).astype(int))
    norm_magz_third_dir = glob_results_dir + \
                          "PiFluxGutzwiller_8_28/Dirac_finite_Lx_#_Ly_6_chi_10000_flux_0.0_YC_gsindex_0_magz_0.3330_monQ_#_svdmin_0.0005/"
    AnalyzeMagnetizedJ1J2Correlations(norm_magz_third_dir, ns, postprocess_dir=False, Ly=Ly,
                                      monQs=(ns/Ly).astype(int))

    plt.show()
    exit(0)

    output_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Meetings/4_5_2026/"
    monopole_dir = glob_results_dir + "MonopoleCondensateGutzwiller/"

    gutzwiller_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Code/MonopoleCondensateGutzwiller/" + \
                      "Dirac_finite_Lx_6_Ly_6_chi_2000_flux_0.0_YC_gsindex_0_magz_0.333_monQ_6/"

    #ComputeCorrelationsFromMPSFile(gutzwiller_dir, 6, 6, ("open", "periodic"),
    #                               psi_fname="psi_gutzwiller.pkl", plot_mode="voronoi",
    #                               transverse_correlations=True)

    # results_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Code/LocalJ1J2TriangularDMRGResults/Lx_3_Ly_3_bc_pp_YC/infinite_init_Random_conserve_True_J2_0.0/"
    #mat = np.loadtxt(results_dir + special_points_structure_factor.csv)
    #print(mat)
    #exit(1)

    #PlotCorrelationsFromFiles(results_dir, show_energies=False, plot_mode="voronoi")
    #plt.show()
    #exit(1)

    #getEnergyDifferenceBetweenSectors(project_dir + "Meetings/1_6_2026/XC8/Flux0_Random/",
    #                                  project_dir + "Meetings/1_6_2026/XC8/Flux1_Random/", r"Flux 0 vs. Flux $\pi$",
    #                                  False, "ener_diff_XC8_gutz.png")

    #case_dir = "Meetings/1_6_2026/XC8/Flux0_Random/"
    #PlotCorrelationsFromFiles(project_dir + case_dir, show_energies=False,
    #                          output_dir=project_dir + case_dir, fig_title="gutz")
    #case_dir = "Meetings/1_6_2026/XC8/Flux1_Random/"
    #PlotCorrelationsFromFiles(project_dir + case_dir, show_energies=False,
    #                          output_dir=results_dir + case_dir, fig_title="gutz")
    # calculateZ2EntanglementEntropy()
    # TestDimerDimerCorrelations()
    # PlotSquareLatticeStructureFactor(Lx=3, Ly=3)

    #dir = results_dir + "/LocalGutzwillerResults/Z2_infinite_Lx_2_Ly_4_chi_500_flux_0.0_YC_gsindex_0_magz_1/"
    #with open(dir + "psi_gutzwiller.pkl", 'rb') as f:
    #   psi = pickle.load(f)
    #print(np.sum(psi.expectation_value("Sz")))

    #gutz_dir = results_dir + "LocalGutzwillerResults/"
    #Lx, Ly = 2, 6
    #chi_gutz = 2000
    #flux_gutz = 0.0
    #geometry = "YC"
    #J2 = 0.125
    #calculateGutzwillerEnergyTriangularJ1J2(gutz_dir, Lx, Ly, chi_gutz, flux_gutz, geometry=geometry,
    #                                        J2=J2, Delz=1.0, bc_MPS="infinite", bc=("periodic", "periodic"))
    #

    #arr = np.array([[1+1j, 1-1j], [1-1j, 1-1j]])
    #print(det(arr))
    #exit(0)

    # C_magnetized = np.loadtxt("debug_magnetized_iMPS/C_Lx_80_m_0.0833.txt", dtype=np.complex128)
    # C_unmagnetized = np.loadtxt("debug_magnetized_iMPS/C_Lx_80_m_0.0.txt", dtype=np.complex128)
    # fig, ax = plt.subplots(figsize=(5,6))
    # # ImshowMatrix(ax, fig, np.array([0.0,1.0]), np.array([0.0,1.0]), np.abs(C), xlabel="X", ylabel="Y")
    # middle_site = C_magnetized.shape[0]//2
    # ax.plot(C_magnetized[middle_site, (middle_site)::2], "bo", markersize=2, label="with magz")
    # ax.plot(C_unmagnetized[middle_site, (middle_site)::2], "ro", markersize=2, label="without magz")
    # ax.legend()
    # plt.show()


    # calculateOverlapsFastLocal(6, 6, 2, 0.056)

    #ComputeCorrelationsFromMPSFile(results_dir, 6, 6, ("open", "periodic"), "finite", geometry="YC",
    #                               psi_dir="LocalGutzwillerResults/Dirac_finite_Lx_6_Ly_6_chi_1000_flux_0.0_YC_gsindex_0_magz_6_monQ_8/",
    #                               psi_fname="psi_gutzwiller.pkl", transverse_correlations=True)
    #plt.show()

    #plotMonopoleOrderParameter()
    #plt.show()

    #Lx, Ly = 6, 6
    # magz = 0.111
    # for Q in [0,2]:
    #     dir = monopole_dir + f"/Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz}_monQ_{Q}/"
    #bc = ("open", "periodic")
    #site = SpinHalfSite('Sz')
    #lat =  BuildTriangularLattice(Lx, Ly, site, "finite", bc=bc)
    #     with open(dir + "psi_gutzwiller.pkl", 'rb') as f:
    #         psi = pickle.load(f)
    #_, fig, ax = plot_scalar_spin_chirality(psi3, lat)
    #fig.savefig(meetings_dir + f"20_6_2026/chirality_magz_{0.056}_J2_0.1_dmrg.png", bbox_inches='tight')
    #plt.show()

    #dir = results_dir + "LocalGutzwillerResults/Dirac_finite_Lx_6_Ly_6_chi_1000_flux_0. 0_YC_gsindex_0_magz_6_monQ_6/"
    #with open(dir + "psi_gutzwiller.pkl", 'rb') as f:
    #  psi = pickle.load(f)

    #calculateGutzwillerEnergyTriangularJ1J2("LocalGutzwillerResults/", 2, 8, 2500, 0.0,
    #                                        "infinite", 0.125, 1.0, ("periodic", "periodic"), "YC",
    #                                        0)

    # spin_corr_x1 = np.loadtxt(psi1_dir + "spin_corr_x.csv")
    # spin_corr_x2 = np.loadtxt(psi2_dir + "spin_corr_x.csv")
    # plt.imshow(spin_corr_x1 - spin_corr_x2)
    # plt.show()
    #
    # psi_fname = "psi_gutzwiller.pkl"
    # with open(psi1_dir + psi_fname, 'rb') as f:
    #     psi1 = pickle.load(f)
    # with open(psi2_dir + psi_fname, 'rb') as f:
    #     psi2 = pickle.load(f)
    # print(psi1.overlap(psi2))
