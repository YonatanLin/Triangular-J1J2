from TryingTemfpy import local


import matplotlib.pyplot as plt
from TryingTemfpy import rc_params
plt.rcParams.update(rc_params)

from temfpy import slater
import temfpy.gutzwiller as gutz
import tenpy
import numpy as np
from numpy import log, sin, cos, sqrt, pi
from tenpy.models.model import CouplingModel, CouplingMPOModel, Lattice
from tenpy.networks.mps import MPS

import tenpy.linalg.np_conserved as npc
from tenpy.models import lattice
from tenpy.algorithms import dmrg
from numpy.linalg import norm, eigh
from tenpy.tools.misc import setup_logging
import pickle
from tenpy.networks.site import FermionSite, SpinHalfSite
from tenpy.models.spins import SpinModel
from pathlib import Path
import json

setup_logging(to_stdout="INFO")

default_chi_max = 3000
default_dmrg_params = {'mixer': True, 'max_E_err': 1.0e-10, 'trunc_params': {'chi_max': default_chi_max, 'svd_min': 1.0e-7},
                   'combine': True, 'chi_list': {0: 50, 3: 100, 7: default_chi_max}, 'min_sweeps': 10, 'max_sweeps': 30,
                   'N_sweeps_check': 1}
code_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Code/"


def ChangeChiInDMRGParams(chi_max):
    dmrg_params_copy = {key: default_dmrg_params[key] for key in default_dmrg_params.keys()}
    chi_max = 500
    dmrg_params_copy["trunc_params"]["chi_max"] = chi_max
    dmrg_params_copy["max_sweeps"] = 20
    dmrg_params_copy["chi_list"] = {0: 50, 3: 100, 7: chi_max}
    return dmrg_params_copy

def AddAndTrackCoupling(model, strength, u1, op1, u2, op2, dx, couplings_list, plus_hc=False):
    model.add_coupling(strength, u1, op1, u2, op2, dx, plus_hc=plus_hc)
    couplings_list.append((u1, u2, dx))


def PrintCouplings(model, Lx, Ly, fancy_print=False):
    print("couplings: ")
    couplings_list = model.all_coupling_terms().to_TermList()
    if fancy_print:
        for i in range(Lx * Ly):
            for j in range(i, Lx * Ly):
                for coupling in couplings_list:
                    if (coupling[0][0][1] == i and coupling[0][1][1] == j):
                        print(coupling)
    else:
        print(couplings_list)

def CreateHamiltonianMatrixFromCouplingsList(couplings_list, N_sites):
    H = np.zeros((N_sites, N_sites))
    i = 0
    for coupling in couplings_list:
        i += 1
    for coupling in couplings_list:
        strength = coupling[1]
        site1 = coupling[0][0][1]
        site2 = coupling[0][1][1]
        if "Cd" in coupling[0][0][0]:
            H[site1, site2] = strength
        else:
            H[site2, site1] = strength
    assert (np.abs(H - np.conj(np.transpose(H))) < 1e-15).all()
    return H

def CalculateSpinSpinCorrelations(psi, sites1=None, sites2=None, inf_mps_unitcell_fac=3):
    if psi.bc == "infinite" and sites1 is None and sites2 is None:
        L = psi.L
        sites1 = np.arange(0, inf_mps_unitcell_fac*L)
        sites2 = np.arange(0, inf_mps_unitcell_fac*L)

    pm_corr = psi.correlation_function("Sp", "Sm", sites1=sites1, sites2=sites2)
    mp_corr = psi.correlation_function("Sm", "Sp", sites1=sites1, sites2=sites2)
    zz_corr = psi.correlation_function("Sz", "Sz", sites1=sites1, sites2=sites2)
    spin_corr_x = 0.5 * (pm_corr + mp_corr) + zz_corr
    return spin_corr_x


def PlotLattice(lat, ax, additional_couplings_to_plot=None, plot_nn_couplings=True, nnn_color="green",
                nnn_line_style="-", plot_order=True):
    #if add_nn_explicitly:
    #    lat.plot_coupling(ax, coupling=nn_couplings_list, linewidth=1.0)
    if plot_nn_couplings:
        lat.plot_coupling(ax, linewidth=1.0)

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

def ComputeMomentumSpaceStructureFactorSymmetrized(corr_x, lat, assert_realness=True):
    corr_x_shape = corr_x.shape
    kx = ky = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    Kx, Ky = np.meshgrid(kx, ky)
    corr_k = np.zeros(Kx.shape, dtype=complex)

    unit_cell_positions = lat.unit_cell_positions
    basis_vectors = np.asarray(lat.basis, dtype=float)
    for i in range(corr_x_shape[0]):
        coor_i = lat.mps2lat_idx(i)
        pos_i = (np.dot(coor_i[:2], basis_vectors) + unit_cell_positions[coor_i[-1],:])
        for j in range(corr_x_shape[1]):
            coor_j = lat.mps2lat_idx(j)
            pos_j = (np.dot(coor_j[:2], basis_vectors) + unit_cell_positions[coor_j[-1], :])
            r_ij = pos_i - pos_j
            phases = np.exp(-1j*(Kx * r_ij[0] + Ky * r_ij[1]))
            corr_k += corr_x[i, j] * phases
    corr_k = corr_k / lat.N_sites
    if assert_realness:
        assert(np.max(np.abs(np.imag(corr_k))) < 1e-13)

    return Kx, Ky, corr_k


def ComputeMomentumSpaceStructureFactor(corr_x, lat, assert_realness=True):
    Lx, Ly = lat.Ls[0], lat.Ls[1]
    kx = ky = np.linspace(-2*np.pi, 2*np.pi, 100)
    Kx, Ky = np.meshgrid(kx, ky)
    lat_basis = lat.basis
    unit_cell_positions = lat.unit_cell_positions
    center_site_coordinates = [Lx // 2, Ly // 2, len(unit_cell_positions) // 2]
    center_site_mps_index = lat.lat2mps_idx(center_site_coordinates)
    print("center site index for calculating correlations: ", center_site_mps_index)

    basis_vectors = np.asarray(lat_basis, dtype=float)
    center_site_loc = (np.dot(center_site_coordinates[:2], basis_vectors) +
                       unit_cell_positions[center_site_coordinates[-1],:])
    print("center site index for calculating correlations: ", center_site_mps_index)
    spin_corr_with_center = corr_x[center_site_mps_index, :]
    site_coordinates = np.asarray(lat.order[:, :2], dtype=float)
    site_locations = site_coordinates @ basis_vectors

    for i in range(site_locations.shape[0]):
        unitcell_index = lat.mps2lat_idx(i)[-1]
        site_locations[i,:] += unit_cell_positions[unitcell_index, :]

    r_from_center = site_locations - center_site_loc
    phases = np.exp(-1j * (Kx[..., np.newaxis] * r_from_center[:, 0] +
                          Ky[..., np.newaxis] * r_from_center[:, 1]))
    spin_corr_k = np.sum(spin_corr_with_center[np.newaxis, np.newaxis, :] * phases, axis=2) / lat.N_sites

    if assert_realness:
        assert(np.max(np.abs(np.imag(spin_corr_k))) < 1e-13)
    return Kx, Ky, spin_corr_k


def PlotSquareLatticeStructureFactor(Lx=6, Ly=6):
    site = SpinHalfSite(conserve=None)
    square_lat = lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=['open', 'open'])

    fig_lat, ax_lat = plt.subplots()
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
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr_x, square_lat)

    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
    ImshowMatrix(ax, fig, Kx, Ky, spin_corr_k, title=title)

    square_lat.plot_brillouin_zone(ax)

    fig.tight_layout()
    plt.show()


def Generate120DegOrderedState(lat=None, Lx=None, Ly=None, plot=False):
    if lat == None:
        site = SpinHalfSite(conserve=None)
        Lx = 9
        Ly = 9
        triangular_lat = lattice.Triangular(Lx=Lx, Ly=Ly, site=site, bc=['periodic', 'open'])
        lat = triangular_lat
    else:
        site = lat.mps_sites()[0]
    basis = lat.basis
    unitcell_pos = lat.unit_cell_positions

    aligned_with_x = (basis[0][1] == 0.0 or basis[1][1] == 0.0)
    aligned_with_y = (basis[0][0] == 0.0 or basis[1][0] == 0.0)
    assert(aligned_with_x or aligned_with_y)

    psi = MPS.from_product_state(lat.mps_sites(), ["up"] * lat.N_sites)

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
        fig_lat, ax_lat = plt.subplots()
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        spin_corr_x = CalculateSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr_x, lat)
        fig_corr, ax_corr = plt.subplots()
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k)
        lat.plot_brillouin_zone(ax_corr)
        plt.show()

    return psi


def GenerateStripeOrderedState(lat, plot=False):
    site = lat.mps_sites()[0]
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
        fig_lat, ax_lat = plt.subplots()
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        plt.show()

    return psi


def TestCorrelationsWithNontrivialUnitCell(state="120"):
    site = SpinHalfSite(conserve=None)
    unit_cell_spin_lat = [[0.0, 0.0], [1.0, 0.0]]
    basis = [[2.0, 0.0], [0.5, sqrt(3) / 2.]]
    Lx, Ly = 4, 5
    triangular_lat_enlarged = BuildTriangularLatticeAlignedWithX(Lx, Ly, site, "finite", unit_cell=unit_cell_spin_lat,
                                                                 basis=basis)

    if state == "120":
        spin_state = Generate120DegOrderedState(triangular_lat_enlarged, Lx, Ly, False)
    elif state == "stripe":
        spin_state = GenerateStripeOrderedState(triangular_lat_enlarged, False)
    else:
        ValueError("Illegal spin state option")
        return

    print(spin_state.expectation_value("Sz"))

    fig_lat, ax_lat = plt.subplots()
    PlotLattice(triangular_lat_enlarged, ax_lat, plot_order=True)
    plt.show()

    spin_corr_x = CalculateSpinSpinCorrelations(spin_state)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr_x, triangular_lat_enlarged)
    fig, ax = plt.subplots()
    ImshowMatrix(ax, fig, Kx, Ky, np.abs(spin_corr_k))
    triangular_lat = BuildTriangularLatticeAlignedWithX(Lx, Ly, site, "finite")
    triangular_lat.plot_brillouin_zone(ax)
    plt.show()


def RunDMRG(model, psi_init, dmrg_params=default_dmrg_params,
            plot_convergence=True, print_final_results=False,
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
        fig,ax = plt.subplots()
        ax.plot([-1] + sweeps, [E_initial] + energies, "o")
        ax.set_title("DMRG Sweeps Energies")
        ax.set_xlabel("sweep")
        ax.set_ylabel("E")
        if expected_energy is not None:
            ax.axhline(expected_energy, color="red", linewidth=0.6, alpha=0.5, linestyle="dashed")
        if energies_fig_title is not None:
            fig.savefig(results_dir + energies_fig_title, bbox_inches='tight')

        plt.show()
    return E_initial, E_final, sweeps, energies, info


def TestSquareLattice(Lx=5, Ly=5, bc=('open', 'open'), J2s=[0.0],
                      bc_MPS="finite"):
    for J2 in J2s:
        site = SpinHalfSite(conserve='Sz')
        square_lat = lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=list(bc), bc_MPS=bc_MPS)
        J = 1.0

        J1J2_model = SpinModel({"lattice": square_lat, "Jx": J, "Jy": J, "Jz": J})

        nnn_couplings_list = []
        if abs(J2) > 0.0:
            for basis_vec in ([-1, -1], [-1, 1], [1, -1], [1, 1]):
                AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sp", 0, "Sm", basis_vec,
                                    nnn_couplings_list)
                AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sm", 0, "Sp", basis_vec,
                                    nnn_couplings_list)
                AddAndTrackCoupling(J1J2_model, J2, 0, "Sz", 0, "Sz", basis_vec,
                                    nnn_couplings_list)

        J1J2_model.init_H_from_terms()

        results_dir = "SquareLatticeJ1J2/"
        fig_lat, ax_lat = plt.subplots()
        PlotLattice(square_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)
        fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')

        psi = MPS.from_product_state(
            square_lat.mps_sites(),
            ["up"] * (square_lat.N_sites // 2) + ["down"] * (square_lat.N_sites - square_lat.N_sites // 2),
            bc=square_lat.bc_MPS,
            unit_cell_width=square_lat.mps_unit_cell_width,
        )
        psi.canonical_form()

        initial_energy = J1J2_model.H_MPO.expectation_value(psi)
        initial_energy_per_site = initial_energy
        if bc_MPS == "finite":
            initial_energy_per_site /= square_lat.N_sites
        #if J2 == 0.0:
        #    assert(initial_energy_per_site == 0.5 * (1. - 1. / L))
        print("Initial energy per site: ", initial_energy_per_site)
        energies_fig_title = "energy_convergence_J2_" + str(J2) + ".png"
        E_initial, E_gs, sweeps, energies, info = RunDMRG(J1J2_model, psi,
                                                          print_final_results=True, results_dir=results_dir,
                                                          energies_fig_title=energies_fig_title)
        energy_per_site = E_gs

        with open("Energy_J2_"+str(J2)+".txt", "w") as f:
            f.write(f"Energy per site = {energy_per_site:.13f}")

        if bc_MPS == "finite":
            energy_per_site /= square_lat.N_sites
        print(f"Energy per site = {energy_per_site:.13f}")

        with open(results_dir + 'psi_gs_J2_'+str(J2)+".pkl", 'wb') as f:
            pickle.dump(psi, f)

        spin_corr = CalculateSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr, square_lat,
                                                                  assert_realness=False)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        title = f"Spin structure factor"
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k, title=title)
        square_lat.plot_brillouin_zone(ax_corr)
        fig_corr.tight_layout()

        fig_corr.savefig("SquareLatticeJ1J2/spin_correlations_J2_"+str(J2)+".png", bbox_inches='tight')

        plt.show()
    # return energy_per_site, psi, square_lat


def BuildTriangularLatticeAlignedWithX(Lx, Ly, site, bc_MPS,
                                       bc = ('periodic', 'periodic'), unit_cell = None, basis=None):
    if basis is None:
        basis = [[1.0, 0.0], [0.5, sqrt(3) / 2.]]

    using_default_unit_cell = unit_cell is None
    if using_default_unit_cell:
        unit_cell = [[0.0, 0.0]]
    nearest_neighbors = []
    if using_default_unit_cell:
        nearest_neighbors = [[0, 0, [1, 0]], [0, 0, [0, 1]], [0, 0, [1, -1]]]

    triangular_lat = lattice.Lattice([Lx, Ly], [site]*len(unit_cell), basis=basis,
                                     positions=unit_cell, bc=bc, pairs={'nearest_neighbors': nearest_neighbors},
                                     bc_MPS=bc_MPS)
    return triangular_lat


def GetTriangularLatticeInitialState(initial_state, triangular_lat, magz=0):
    Lx = triangular_lat.Ls[0]
    Ly = triangular_lat.Ls[1]
    N_sites = triangular_lat.N_sites
    if initial_state == "Random":
        N_up = N_sites // 2 + magz
        N_down = N_sites - N_up
        down_indices = np.random.choice(np.arange(0, N_sites), N_down, replace=False)
        print(down_indices)
        down_indices = np.array(down_indices)
        product_state = ["up"] * N_sites
        for down_ind in down_indices:
            product_state[down_ind] = "down"
        psi = MPS.from_product_state(triangular_lat.mps_sites(), product_state, bc=triangular_lat.bc_MPS)

    elif initial_state == "120":
        psi = Generate120DegOrderedState(lat=triangular_lat, Lx=Lx, Ly=Ly)

    elif initial_state == "stripe":
        psi = GenerateStripeOrderedState(lat=triangular_lat)

    else:
        ValueError("unrecognized initial state")
        psi = None
    return psi


def TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, flux, initial_state, conserve, J2):
    bc_string = ""
    for bc_ax in bc:
        if bc_ax == "periodic":
            bc_string += "p"
        else:
            assert(bc_ax == "open")
            bc_string += "o"

    geometry_dir = f"Lx_{Lx}_Ly_{Ly}_bc_{bc_string}/"
    params_dir = f"mps_{bc_MPS}_flux_{flux}_init_{initial_state}_conserve_{conserve}_J2_{J2}/"
    return geometry_dir, params_dir


def CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc, bc_MPS, flux, initial_state, conserve, J2):
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    geometry_dir, params_dir = TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, flux, initial_state, conserve, J2)
    results_dir = main_results_dir + geometry_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_dir += params_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir


def CreateTriangularCaseDirFromInputFile(main_results_dir, input_file):
    with open(input_file, "r") as file:
        input_file_lines = file.readlines()
    params = input_file_lines[0].split(" ")
    print(params)
    assert(params[0] == "Lx" and params[1] == "Ly" and params[2] == "bc" and params[3] == "bc_MPS"
           and params[4] == "flux" and params[5] == "initial_state" and params[6] == "conserve" and params[7] == "J2\n")
    input_for_condor = open("condor_cases.txt", 'w')
    for line in input_file_lines[1:]:
        params = line.split(" ")
        case_folder = CreateTriangularCaseDir(main_results_dir, params[0], params[1], params[2].split("-"),
                params[3], params[4], params[5], params[6], params[7][:-1])
        input_for_condor.write(line[:-1] + " " + case_folder + "\n")


def GenerateJ1J2SpinTriangularModel(J2, triangular_lat):
    J1 = 1.0
    J1J2_model = SpinModel({"lattice": triangular_lat, "Jx": J1, "Jy": J1, "Jz": J1})
    J1J2_model.manually_call_init_H = True

    nnn_couplings_list = []
    if abs(J2) > 0.0:
        for basis_vec in ([1, 1], [-1, 2], [-2, 1]):
            AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sp", 0, "Sm", basis_vec,
                                nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sm", 0, "Sp", basis_vec,
                                nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, J2, 0, "Sz", 0, "Sz", basis_vec,
                                nnn_couplings_list)
    J1J2_model.init_H_from_terms()
    return J1J2_model


def BuildSpinTriangularLatticeWithGutzwillerOrdering(Lx, Ly, site, bc_MPS, bc):
    # As the pi flux has a doubled unit cell, the ordering groups pairs along the x-axis, then traverses along
    # y direction in a snake form.
    triangular_lat = BuildTriangularLatticeAlignedWithX(Lx, Ly, site, bc_MPS, bc=bc)
    lat_order = []
    for i in range(Lx // 2):
        for j in range(Ly):
            lat_order += [(2 * i, j, 0), (2 * i + 1, j, 0)]
    triangular_lat.order = np.array(lat_order)
    return triangular_lat
    
    
def calculateGutzwillerEnergyTriangularJ1J2(Lx, Ly, J2=0.125,
                            bc_MPS="finite", bc=("open", "periodic"), reorder_lattice=True):
    Lx_gutz = int(Lx // 2)
    #psi_path = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Code" + \
    #           f"/TriangularPiFluxGutzwiller/Lx_{Lx_gutz}_Ly_{Ly}_bc_op/psi_gutzwiller.pkl"
    psi_path = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Code" + \
               f"/TriangularPiFluxGutzwiller/Lx_{Lx_gutz}_Ly_{Ly}_chi_6000/psi_gutzwiller.pkl"
    site = SpinHalfSite(conserve="Sz")

    with open(psi_path, 'rb') as f:
        psi = pickle.load(f)
        psi.unit_cell_width = 1
    print(psi.L)

    if reorder_lattice:
        #triangular_lat = BuildSpinTriangularLatticeWithGutzwillerOrdering(Lx, Ly, site, bc_MPS, bc)
        #J1J2_model = GenerateJ1J2SpinTriangularModel(J2, triangular_lat)
        triangular_lat = BuildTriangularLatticeAlignedWithX(Lx, Ly, site, bc_MPS, bc)
        J1J2_model = GenerateJ1J2SpinTriangularModel(J2, triangular_lat)
        psi_copy = psi.copy()
        PermuteGutzwillerWavefunctionToDMRGOrder(psi, Lx_gutz, Ly)
        print("orig overlap with permutation: ", psi.overlap(psi_copy))
        print("permutation norm squared: ", psi.overlap(psi))

    fig, ax = plt.subplots()
    PlotLattice(triangular_lat, ax)
    plt.show()
    print(triangular_lat.N_sites)
    E = J1J2_model.H_MPO.expectation_value(psi)
    if bc_MPS == "finite":
        E /= triangular_lat.N_sites
    print("Energy: ", E)


def TriangularJ1J2DMRG(Lx, Ly, bc, bc_MPS, flux=0.0, conserve=True,
                       initial_state="Random", J2=0.0, basis=None, unit_cell=None):
    if isinstance(bc, str):
        bc_parsed = bc.split("-")
        bc = (bc_parsed[0], bc_parsed[1])

    dmrg_params = default_dmrg_params
    if local:
        main_results_dir = "TriangularLatticeResults/"
        results_dir = CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc, bc_MPS, flux, initial_state, conserve, J2)
    else:
        results_dir = "./"

    with open(results_dir + "dmrg_params.json", "w") as f:
        json.dump(dmrg_params, f, indent=4)


    if conserve:
        site = SpinHalfSite(conserve='Sz')
    else:
        site = SpinHalfSite(conserve=None)

    triangular_lat = BuildTriangularLatticeAlignedWithX(Lx, Ly, site, bc_MPS, bc=bc)

    center_site_mps_index = triangular_lat.lat2mps_idx([Lx // 2, Ly // 2, 0])
    print("center site mps index: ", center_site_mps_index)

    J1J2_model = GenerateJ1J2SpinTriangularModel(J2, triangular_lat)


    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    PlotLattice(triangular_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)
    fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')

    print_couplings = False
    if print_couplings:
        PrintCouplings(J1J2_model, Lx, Ly)

    psi = GetTriangularLatticeInitialState(initial_state, triangular_lat)

    with open(results_dir + 'psi_initial' + ".pkl", 'wb') as f:
        pickle.dump(psi, f)

    RunDMRG(J1J2_model, psi, dmrg_params=dmrg_params, print_final_results=True, results_dir=results_dir,
            energies_fig_title="energies.png")

    with open(results_dir + 'psi_gs' + ".pkl", 'wb') as f:
        pickle.dump(psi, f)

    sites1, sites2 = None, None
    lat_for_corr = triangular_lat
    if bc_MPS == "infinite":
        Lx_large = 10 * Lx
        sites1 = np.arange(0, Ly * Lx_large)
        sites2 = np.arange(0, Ly * Lx_large)
        lat_for_corr = BuildTriangularLatticeAlignedWithX(Lx_large, Ly, site, bc_MPS)

    spin_corr_x = CalculateSpinSpinCorrelations(psi, sites1, sites2)
    np.savetxt("spin_corr_x.csv", spin_corr_x)

    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr_x, lat_for_corr,
                                                                         assert_realness=False)
    np.savetxt("Kx.csv", Kx)
    np.savetxt("Ky.csv", Ky)
    np.savetxt("spin_corr_k.csv", spin_corr_k)

    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
    ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k, title=title)
    #lat_for_corr.plot_brillouin_zone(ax_corr)
    fig_corr.savefig(results_dir + "momentum_space_correlations.png",
                     bbox_inches='tight')


def DeterminePiFluxCoupling(x, y, dx, dy, basis_vectors):
    assert((basis_vectors[0][:] == [1.0, 0.0]).all())
    assert ((np.abs(basis_vectors[1][:] - [0.5, sqrt(3) / 2.]) < 1e-15).all())
    if dx == 1 and dy == 0:
        return 1.0
    elif dx == 0 and dy == 1:
        if x % 2 == 0:
            return -1.0
        return 1.0
    else:
        assert(dx == -1 and dy == 1)
        if x % 2 == 0:
            return 1.0
        return -1.0


class FermionicPiFluxModel(CouplingMPOModel):
    def init_terms(self, model_params):
        neg_hoppings_list = []
        pos_hoppings_list = []
        plus_hc = True
        rec_long_side_coors = model_params["rec_long_side_coors"]
        spinfull = model_params["spinfull"]
        ph = model_params["particle_hole"]
        init_MPO = model_params["init_H_MPO"]
        if spinfull:
            unitcell_hopping_pairs = [(0,2), (1,3)]
        else:
            unitcell_hopping_pairs = [(0, 1)]
        for pair in unitcell_hopping_pairs:
            s1 = pair[0]
            s2 = pair[1]
            sgn = 1.0
            if ph and s1 == 1:
                sgn = -1.0
            pos_coupling = sgn
            neg_coupling = -1.0 * sgn
            AddAndTrackCoupling(self, pos_coupling, s2, "Cd", s2, "C", [0,-1], pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, neg_coupling, s1, "Cd", s1, "C", [0,-1], neg_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, pos_coupling, s2, "Cd", s1, "C", [0, 0], pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, neg_coupling, s2, "Cd", s1, "C", [0,1], neg_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, pos_coupling, s1, "Cd", s2, "C", np.array([0,-1]) - np.array(rec_long_side_coors),
                                pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, pos_coupling, s1, "Cd", s2, "C", (-1) * np.array(rec_long_side_coors),
                                pos_hoppings_list, plus_hc=plus_hc)

        self.neg_hoppings_list = neg_hoppings_list
        self.pos_hoppings_list = pos_hoppings_list
        if init_MPO:
            self.init_H_from_terms()


def GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, bc_MPS):
    lat_basis = [[2.0, 0], [0.5, sqrt(3) / 2.]]
    if spinfull:
        unitcell_pos = [[-0.1, 0.0], [0.1, 0.0], [0.9, 0.0], [1.1, 0.0]]
    else:
        unitcell_pos = [[0.0, 0.0], [1.0, 0.0]]

    rec_long_side_coors = [1, -1]
    # nearest_neighbors = [(1, 1, [1, 0]), (0, 0, [1, 0]), (0,1,[0,0]), (0,1,[-1,0]),
    #                      (1,0,[0,1]), (1,0,[-1,1])]
    nearest_neighbors = []
    bc = ['open', 'periodic']
    # bc = ['periodic', -1]
    if bc_MPS == "infinite":
        bc[0] = 'periodic'
    pairs = {'nearest_neighbors': nearest_neighbors}
    ordering = []
    for i in range(Lx):
        for j in range(Ly):
            ordering.append((i, j, 0))
            ordering.append((i, j, 1))
        for j in range(Ly):
            ordering.append((i, j, 2))
            ordering.append((i, j, 3))

    triangular_lat = lattice.Lattice([Lx, Ly], [site] * len(unitcell_pos), basis=lat_basis,
                                     positions=unitcell_pos, bc=bc, pairs=pairs, bc_MPS=bc_MPS, order=order)
    return triangular_lat, {"rec_long_side_coors": rec_long_side_coors, "unitcell_pos":unitcell_pos}


def CalculateExactCMatrixForPiFlux(Lx, Ly, spinfull, site, zero_energy_tol=1e-9,
                                   particle_hole=True, plot_lattice=False):
    bc_slater = ["open", "periodic"]
    triangular_lat, params = GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, "finite")

    rec_long_side_coors = params["rec_long_side_coors"]
    pi_flux_model = FermionicPiFluxModel({"lattice": triangular_lat,
                                          "rec_long_side_coors": rec_long_side_coors,
                                          "spinfull": spinfull, "particle_hole":particle_hole,
                                          "init_H_MPO": False})
    if plot_lattice:
        fig, ax = plt.subplots()
        PlotLattice(triangular_lat, ax, additional_couplings_to_plot=pi_flux_model.pos_hoppings_list)
        PlotLattice(triangular_lat, ax, additional_couplings_to_plot=pi_flux_model.neg_hoppings_list,
                    nnn_line_style="--", )
        plt.show()

    couplings_list = pi_flux_model.all_coupling_terms().to_TermList()
    H = CreateHamiltonianMatrixFromCouplingsList(couplings_list, triangular_lat.N_sites)
    e, v = eigh(H)
    if np.min(np.abs(e)) < zero_energy_tol:
        print("cant handle zero modes in Gutzwiller projections!")
        exit(1)

    C, _ = slater.correlation_matrix(H, triangular_lat.N_sites // 2)
    return C, triangular_lat


def GetTriangularFluxSlaterMPS(Lx, Ly, spinfull, site, mps_unitcell, slater_trunc_par,
                               finite = True, particle_hole=True):
    zero_energy_tol = 1e3 * slater_trunc_par["degeneracy_tol"]
    C = None
    if finite:
        C, triangular_lattice = CalculateExactCMatrixForPiFlux(Lx, Ly, spinfull, site,
                                                               zero_energy_tol = zero_energy_tol,
                                                               particle_hole=particle_hole,
                                                               plot_lattice=False)

        psi_from_slater = slater.C_to_MPS(C, trunc_par=slater_trunc_par)
    else:
        Lx_short, Lx_long = Lx, Lx + 1
        C_short, triangular_lat_short = CalculateExactCMatrixForPiFlux(Lx_short, Ly, spinfull, site,
                                                                       zero_energy_tol = zero_energy_tol,
                                                                       particle_hole=particle_hole)
        C_long, _ = CalculateExactCMatrixForPiFlux(Lx_long, Ly, spinfull, site, zero_energy_tol = zero_energy_tol,
                                                   particle_hole=particle_hole)

        middle_site_mps_ind_short = triangular_lat_short.lat2mps_idx([Lx_short // 2, 0, 0])

        psi_from_slater, error = slater.C_to_iMPS(C_short, C_long, slater_trunc_par,
                                                  sites_per_cell=mps_unitcell,
                                                  cut=middle_site_mps_ind_short)
    return psi_from_slater, C


def TriangularPiFluxAnsatz(Lx=2, Ly=3, spinfull=True, bc_MPS="infinite", particle_hole=False,
                           chi_max_temfpy = 600):
    #if bc_MPS == "infinite":
    #    assert(Lx == 1)
    main_results_dir = "PiFluxAnsatzResults/"

    site = FermionSite(conserve='N')
    triangular_lat, params_lat = GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, bc_MPS)

    bc_lat = triangular_lat.boundary_conditions
    conserve_N = (site.conserve=='N')
    results_dir = CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc_lat, bc_MPS,
                                      0.0, "Random", conserve_N, 0.0)
    rec_long_side_coors = params_lat["rec_long_side_coors"]
    unitcell_pos = params_lat["unitcell_pos"]

    pi_flux_model = FermionicPiFluxModel({"lattice": triangular_lat, "rec_long_side_coors":rec_long_side_coors,
                                          "spinfull":spinfull, "particle_hole":particle_hole,
                                          "init_H_MPO": True})

    plot_lattice = True
    if plot_lattice:
        fig_lat, ax_lat = plt.subplots()
        plot_order = False
        PlotLattice(triangular_lat, ax_lat, pi_flux_model.pos_hoppings_list, plot_nn_couplings=False, plot_order=plot_order)
        PlotLattice(triangular_lat, ax_lat, pi_flux_model.neg_hoppings_list, plot_nn_couplings=False,
                    nnn_line_style="--", plot_order=plot_order)
        fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')
        plt.show()

    N_sites = triangular_lat.N_sites

    mps_unitcell = len(unitcell_pos) * Ly
    Lx_short_for_iMPS = 100
    slater_trunc_par = {"chi_max": chi_max_temfpy, "svd_min": 1e-7, "degeneracy_tol": 1e-12}
    zero_energy_tol = 1e3 * slater_trunc_par["degeneracy_tol"]
    Lx_exact_C_infinite = 10
    if bc_MPS == "finite":
        psi_from_slater, C = GetTriangularFluxSlaterMPS(Lx, Ly, spinfull, site, mps_unitcell, slater_trunc_par,
                                                        finite=True, particle_hole=particle_hole)
        psi_from_slater.canonical_form()
    else:
        psi_from_slater, _ = GetTriangularFluxSlaterMPS(Lx_short_for_iMPS, Ly, spinfull, site, mps_unitcell,
                                                        slater_trunc_par,  finite=False, particle_hole=particle_hole)
        C, _ = CalculateExactCMatrixForPiFlux(Lx_exact_C_infinite, Ly, spinfull, site,
                                              zero_energy_tol = zero_energy_tol, particle_hole=particle_hole)
        psi_from_slater.canonical_form()

    sites1 = None
    sites2 = None
    if bc_MPS == "infinite":
        sites1 = np.arange(0, Lx_exact_C_infinite * mps_unitcell)
        sites2 = np.arange(0, Lx_exact_C_infinite * mps_unitcell)
    print("psi slater normalization: ", psi_from_slater.overlap(psi_from_slater))
    E_slater_mps = pi_flux_model.H_MPO.expectation_value(psi_from_slater)
    E_slater_mps_per_site = E_slater_mps
    avg_occupation = 0.5
    if bc_MPS == "finite":
        E_slater_mps_per_site /= (triangular_lat.N_sites * avg_occupation)
    print("Energy per-site for mps-slater:", E_slater_mps_per_site)
    mps_slater_corr = psi_from_slater.correlation_function("Cd", "C", sites1=sites1, sites2=sites2)
    psi_dmrg = MPS.from_product_state(triangular_lat.mps_sites(),
                                      ["full"] * (N_sites // 2) + ["empty"] * (N_sites // 2),
                                      bc=triangular_lat.bc_MPS)
    chi_max = 500
    dmrg_params = ChangeChiInDMRGParams(chi_max)
    RunDMRG(pi_flux_model, psi_dmrg, plot_convergence=True, print_final_results=True, expected_energy=E_slater_mps,
            dmrg_params=dmrg_params, results_dir=results_dir, energies_fig_title="energies.png")
    dmrg_corr = psi_dmrg.correlation_function("Cd", "C", sites1=sites1, sites2=sites2)

    E_per_site_dmrg = pi_flux_model.H_MPO.expectation_value(psi_dmrg)
    if bc_MPS == "finite":
        E_per_site_dmrg /= (triangular_lat.N_sites * avg_occupation)
    print("Energy for mps-slater:", E_slater_mps)
    print("Energy from dmrg:", E_per_site_dmrg)

    assert(C.shape[0] == C.shape[1])
    X,Y = np.meshgrid(np.arange(0,C.shape[0]),np.arange(0,C.shape[0]))
    fig_slater_corr, ax_slater_corr = plt.subplots()
    fig_mps_slater_corr, ax_mps_slater_corr = plt.subplots()
    fig_dmrg_corr, ax_dmrg_corr = plt.subplots()

    ImshowMatrix(ax_slater_corr, fig_slater_corr, X, Y, C, "i", "j")
    ImshowMatrix(ax_mps_slater_corr, fig_mps_slater_corr, X, Y, mps_slater_corr, "i", "j")
    ImshowMatrix(ax_dmrg_corr, fig_mps_slater_corr, X, Y, dmrg_corr, "i", "j")

    with open(results_dir + 'psi_slater' + ".pkl", 'wb') as f:
        pickle.dump(psi_from_slater, f)
    with open(results_dir + 'psi_dmrg' + ".pkl", 'wb') as f:
        pickle.dump(psi_dmrg, f)

    fig_slater_corr.savefig(results_dir + "slater_exact_correlations.png", bbox_inches='tight')
    fig_mps_slater_corr.savefig(results_dir + "slater_mps_correlations.png", bbox_inches='tight')
    fig_dmrg_corr.savefig(results_dir + "dmrg_correlations.png", bbox_inches='tight')

    print("absolute value of overlap between slater and dmrg: ", np.abs(psi_dmrg.overlap(psi_from_slater)))
    print("correlations max distance between slater-mps and dmrg: ", np.max(np.abs(mps_slater_corr - dmrg_corr)))
    print("correlations max distance between slater-mps and exact slater: ", np.max(np.abs(mps_slater_corr - C)))

    print("slater norm: ", psi_from_slater.norm)
    print("dmrg norm: ", psi_dmrg.norm)


def ExplicitMPSNorm(mps):
    tensors = mps._B if mps.bc == "finite" else [mps.get_B(i) for i in range(mps.L)]
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


def GetGutzwillerToDMRGPermutation(Lx, Ly):
    site = SpinHalfSite()

    lat_doubled = BuildSpinTriangularLatticeWithGutzwillerOrdering(Lx, Ly, site, "finite", ["open", "periodic"])
    lat_snake = BuildTriangularLatticeAlignedWithX(Lx, Ly, site, "finite", ["open", "periodic"])

    assert lat_doubled.N_sites == lat_snake.N_sites
    N = lat_doubled.N_sites

    pos_old = np.array([lat_doubled.position(lat_doubled.mps2lat_idx(i)) for i in range(N)])
    pos_new = np.array([lat_snake.position(lat_snake.mps2lat_idx(i)) for i in range(N)])

    p = np.zeros(N, dtype=int)

    for i in range(N):
        dist = np.linalg.norm(pos_new - pos_old[i], axis=1)
        j = np.argmin(dist)

        if dist[j] > 1e-5:
            raise ValueError(f"Site {i} in the old lattice has no physical match in the new lattice!")

        p[i] = j
    return p


def PermuteGutzwillerWavefunctionToDMRGOrder(psi_gutz, Lx, Ly):
    gutz_perm = GetGutzwillerToDMRGPermutation(Lx, Ly)
    perm_trunc_err = psi_gutz.permute_sites(gutz_perm, trunc_par={"chi_max":3*np.max(psi_gutz.chi)})
    print(f"truncation error from permuting Gutzwiller wavefunction: ", perm_trunc_err)


def calculateOverlapBetweenGutzwillerAndDMRG(dmrg_dir, gutzwiller_dir, old_tenpy_version=False,
                                             psi_gutz_fname='psi_gutzwiller_permuted.pkl'):
    with open(dmrg_dir + 'psi_gs.pkl', 'rb') as f_dmrg:
        psi_dmrg = pickle.load(f_dmrg)
    with open(gutzwiller_dir + psi_gutz_fname, 'rb') as f_gutz:
        psi_gutz = pickle.load(f_gutz)
    psi_dmrg.unit_cell_width = 1
    psi_gutz.unit_cell_width = 1

    if old_tenpy_version:
        for i in range(psi_dmrg.L):
            B = psi_dmrg.get_B(i)
            B_inverted = B[:, ::-1, :]
            psi_dmrg.set_B(i, B_inverted, form=psi_dmrg.form[i])

    print("overlap between wavefunctions: ", psi_dmrg.overlap(psi_gutz))


def TriangularPiFluxGutzwiller(Ly, finite=True, Lx=6, chi_max=3000):
    spinfull = True
    site = FermionSite(conserve='N')
    mps_unitcell = 4 * Ly
    spin_site = SpinHalfSite(conserve='Sz')
    gutzwiller_results_dir = "TriangularPiFluxGutzwiller/"

    particle_hole = True
    debug = False

    slater_trunc_par = {"chi_max": chi_max, "svd_min": 1e-7, "degeneracy_tol": 1e-12}
    psi_from_slater, _ = GetTriangularFluxSlaterMPS(Lx, Ly, spinfull, site, mps_unitcell, slater_trunc_par,
                                                    finite=finite, particle_hole=particle_hole)

    if debug:
        triangular_lat, params = GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, "finite")
        pi_flux_model = FermionicPiFluxModel({"lattice": triangular_lat,
                                              "rec_long_side_coors": params["rec_long_side_coors"],
                                              "spinfull": spinfull, "particle_hole": particle_hole,
                                              "init_H_MPO": True})
        print(f"energy per mode of triangular pi flux gs = "
              f"{pi_flux_model.H_MPO.expectation_value(psi_from_slater) / (0.5 * triangular_lat.N_sites)}")

    psi_from_slater.canonical_form()
    if debug:
        print(f"norm of slater mps: {ExplicitMPSNorm(psi_from_slater)}")
        avg_occ = psi_from_slater.expectation_value(["N"])
        print("occ up: ", np.sum(avg_occ[1::2]))
        print("occ down: ", np.sum(avg_occ[0::2]))

        fig_occ, ax_occ = plt.subplots()
        ax_occ.plot(avg_occ, "o")
        plt.show()

    # Since the norm may be below rounding error for large system, canonization can run into problems,
    # as the singular values get very close to zero in the QR decomposition. We rescale the matrices here,
    # to avoid this problem.
    for i in range(len(psi_from_slater._B)):
        Bi = psi_from_slater._B[i]
        physical_axis_shape = Bi.to_ndarray().shape[1]
        psi_from_slater._B[i] = Bi.scale_axis(1.5*np.ones(physical_axis_shape), axis=1)

    if particle_hole:
        psi_gutzwiller = gutz.abrikosov_ph(psi_from_slater, return_canonical=True)
    else:
        psi_gutzwiller = gutz.abrikosov(psi_from_slater, return_canonical=True)

    if debug:
        psi_slater_grouped_pairs = psi_from_slater.copy()
        psi_slater_grouped_pairs.group_sites(2)
        n_conseq_op = [[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        occ_list = range(len(psi_slater_grouped_pairs.sites))
        for i in range(psi_slater_grouped_pairs.L):
            site_doubled = psi_slater_grouped_pairs.sites[i]
            site_doubled.add_op("combined_occ", n_conseq_op)

        fig_double_occ, ax_double_occ = plt.subplots()
        ax_double_occ.plot(psi_slater_grouped_pairs.expectation_value(["combined_occ"]),
                           "o")
        plt.show()

        occ_expect = psi_slater_grouped_pairs.expectation_value_term([("combined_occ", i) for i in occ_list])
        print(f"occ expect: {occ_expect}")
        norm = ExplicitMPSNorm(psi_gutzwiller)
        print(f"norm of psi_gutzwiller is {norm}")

    results_dir = gutzwiller_results_dir + f"Lx_{Lx}_Ly_{Ly}_chi_{chi_max}/"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(results_dir + 'psi_gutzwiller' + ".pkl", 'wb') as f:
        pickle.dump(psi_gutzwiller, f)

    assert(abs(psi_gutzwiller.overlap(psi_gutzwiller) - 1.0) < 1e-7)

    if debug:
        with open(results_dir + 'psi_gutzwiller_not_canonical' + ".pkl", 'wb') as f:
            pickle.dump(psi_gutzwiller, f)
        print(f"norm of canonical psi_gutzwiller is {psi_gutzwiller.overlap(psi_gutzwiller)}")
        print(f"explicit norm of canonical psi_gutzwiller is {ExplicitMPSNorm(psi_gutzwiller)}")


    if not particle_hole:
        return
    if finite:
        spin_corr_x = CalculateSpinSpinCorrelations(psi_gutzwiller)
    else:
        spin_corr_x = CalculateSpinSpinCorrelations(psi_gutzwiller, np.arange(0, Lx),
                                                    np.arange(0, Lx))

    np.savetxt(results_dir + "spin_corr_x.csv", spin_corr_x)

    spin_lat = BuildTriangularLatticeAlignedWithX(2 * Lx, Ly, spin_site, "finite")
    fig_lat, ax_lat = plt.subplots()
    PlotLattice(spin_lat, ax_lat)
    fig_lat.savefig("spin_lattice.png", bbox_inches='tight')

    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr_x, spin_lat)
    np.savetxt(results_dir + "Kx.csv", Kx)
    np.savetxt(results_dir + "Ky.csv", Ky)
    np.savetxt(results_dir + "spin_corr_k.csv", spin_corr_k)

    fig,ax = plt.subplots()
    ImshowMatrix(ax, fig, Kx, Ky, spin_corr_k)
    spin_lat_singlesite_unitcell = BuildTriangularLatticeAlignedWithX(Lx, Ly, spin_site, "finite")
    spin_lat_singlesite_unitcell.plot_brillouin_zone(ax)
    ax.set_title("Spin Correlations")
    fig.savefig(gutzwiller_results_dir + "spin_correlations.png", bbox_inches='tight')
    plt.show()


def ComputeCorrelationsFromMPSFile(parent_results_path, Lx, Ly, bc, bc_MPS, flux=None, initial_state=None,
                                   conserve=None, J2=None, psi_fname="psi_gs.pkl", sort_charge=False,
                                   Lx_for_infinite_bc_MPS=None, psi_dir=None, from_dmrg=True, from_corr_file=False):
    if psi_dir is None:
        geometry_path, params_path = TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, flux, initial_state, conserve, J2)
        psi_dir = parent_results_path + geometry_path + params_path
    else:
        psi_dir = parent_results_path + psi_dir

    with open(psi_dir + psi_fname, 'rb') as f:
        psi = pickle.load(f)
        psi.unit_cell_width = 1

    site = SpinHalfSite(conserve='Sz',sort_charge=sort_charge)
    Lx_correlations = Lx
    if bc_MPS == "infinite" or Lx_for_infinite_bc_MPS is not None:
        assert(Lx_for_infinite_bc_MPS is not None)
        Lx_correlations = Lx_for_infinite_bc_MPS

    if from_dmrg:
        triangular_lattice = BuildTriangularLatticeAlignedWithX(Lx_correlations, Ly, site, "finite")
    else:
        triangular_lattice = BuildSpinTriangularLatticeWithGutzwillerOrdering(Lx_correlations, Ly, site, "finite",
                                                                             bc)
    if from_corr_file:
        spin_corr_x = np.loadtxt(psi_dir + "spin_corr_x.csv")
    else:
        spin_corr_x = CalculateSpinSpinCorrelations(psi, sites1 = np.arange(0, triangular_lattice.N_sites),
                                                    sites2 = np.arange(0,triangular_lattice.N_sites))
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactorSymmetrized(spin_corr_x, triangular_lattice,
                                                                         assert_realness=True)

    spin_corr_k_from_file = np.loadtxt(psi_dir + "spin_corr_k.csv", dtype=complex)
    print("largest diff: ", np.max(np.abs(spin_corr_k - spin_corr_k_from_file)))
    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor"
    ImshowMatrix(ax, fig, Kx, Ky, spin_corr_k, title=title)
    triangular_lattice.plot_brillouin_zone(ax)
    fig.savefig(psi_dir + "momentum_space_correlations_local", bbox_inches='tight')
    plt.show()


def PlotCorrelationsFromFiles(results_dir, energy_ax=None, initial_state="",
                              show_energies = True, psi_fname="psi_gs.pkl"):
    Kx = np.loadtxt(results_dir + "Kx.csv")
    Ky = np.loadtxt(results_dir + "Ky.csv")
    corr_k = np.loadtxt(results_dir + "spin_corr_k.csv", dtype=np.complex128)
    if show_energies:
        if energy_ax is None:
            fig, energy_ax = plt.subplots()
        energies = np.loadtxt(results_dir + "Energies.txt")
        energy_ax.plot(energies, "o", label=initial_state)

    with open(results_dir + psi_fname, 'rb') as f:
        psi = pickle.load(f)
        psi.unit_cell_width = 1

    print("Magentization of gs: ", psi.get_total_charge(only_physical_legs=True))
    print("Magentization of gs: ", np.sum(psi.expectation_value("Sz")))
    fig, ax = plt.subplots()

    triangular_lat = BuildTriangularLatticeAlignedWithX(4, 4, SpinHalfSite(conserve='Sz'), "finite")
    triangular_lat.plot_brillouin_zone(ax)
    ImshowMatrix(ax, fig, Kx, Ky, np.real(corr_k))
    assert (np.max(np.abs(np.imag(corr_k))) < 1e-14)


def PlotRealSpaceCorrelations(results_dir):
    corr_x = np.loadtxt(results_dir + "spin_corr_x.csv", dtype=np.complex128)
    print(corr_x.shape)
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])
    fig, ax = plt.subplots()
    ImshowMatrix(ax, fig, X, Y, np.real(corr_x), label="real part")
    ax.legend()
    

def GutzwillerDMRGComparisons():
    # main_results_dir = "TriangularPiFluxGutzwiller/"
    # case_dir = "Lx_3_Ly_6_chi_6000/"
    # PlotCorrelationsFromFiles(code_dir + main_results_dir + case_dir, show_energies=False,
    #                          psi_fname="psi_gutzwiller.pkl")
    # plt.show()
    # calculateGutzwillerEnergyTriangularJ1J2(6, 6, J2=0.125)
    # calculateGutzwillerEnergyTriangularJ1J2(8, 6, J2=0.125)
    # calculateGutzwillerEnergyTriangularJ1J2(10, 6, J2=0.125)
    # calculateGutzwillerEnergyTriangularJ1J2(12, 6, J2=0.125)

    #Ls = np.array([6, 8, 10, 12])
    #Es = np.array([-0.48526, -0.48691, -0.48785, -0.48847])
    #plt.plot(1./Ls, Es, "--")
    #plt.xlim(0.0, 0.17)
    #plt.ylim(-0.51, -0.48)
    #plt.show()

    #dmrg_dir = code_dir + \
    #           "TriangularLatticeResults/FromCluster/Lx_12_Ly_5_bc_op/mps_finite_flux_0.0_init_Neel_conserve_1_J2_0.135/"
    #dmrg_dir = code_dir + \
    #          "NewTenpyTriangularLatticeResults/Lx_6_Ly_6_bc_op/mps_finite_flux_0.0_init_Random_conserve_1_J2_0.125/"
    #gutz_dir = code_dir + "TriangularPiFluxGutzwiller/Lx_3_Ly_6_chi_6000/"

    #dmrg_dir = code_dir + \
    #           "TriangularLatticeResults/FromCluster/Lx_12_Ly_5_bc_op/mps_finite_flux_0.0_init_Neel_conserve_1_J2_0.125/"
    dmrg_dir = code_dir + \
               "TriangularLatticeResults/FromCluster/Lx_12_Ly_5_bc_op/mps_finite_flux_0.0_init_stripe_conserve_1_J2_0.125/"
    gutz_dir = code_dir + "TriangularPiFluxGutzwiller/Lx_6_Ly_5_bc_op/"
    calculateOverlapBetweenGutzwillerAndDMRG(dmrg_dir, gutz_dir, 12, 5, old_tenpy_version=False)



def DMRGCorrelations():
    main_results_dir = "NewTenpyTriangularLatticeResults/"
    case_dir = f"Lx_6_Ly_6_bc_op/mps_finite_flux_0.0_init_Random_conserve_1_J2_{0.0}/"
    PlotCorrelationsFromFiles(code_dir + main_results_dir + case_dir)
    plt.show()

    for J2 in [0.1, 0.11, 0.12, 0.125, 0.13, 0.14, 0.15]:
        fig, energy_ax = plt.subplots()
        for initial_state in ["stripe", "Random"]:
            case_dir = f"Lx_6_Ly_6_bc_op/mps_finite_flux_0.0_init_{initial_state}_conserve_1_J2_{J2}/"
            PlotCorrelationsFromFiles(code_dir + main_results_dir + case_dir, energy_ax=energy_ax,
                                      initial_state=initial_state)
        energy_ax.legend()
        plt.show()


def PermuteGutzwillerResults():
    main_dir = code_dir + "TriangularPiFluxGutzwiller/"
    # cases = ["Lx_3_Ly_6_chi_6000/", "Lx_4_Ly_6_chi_6000/", "Lx_5_Ly_6_chi_6000/", "Lx_6_Ly_6_chi_6000/"]
    cases = ["Lx_6_Ly_4_bc_op/", "Lx_6_Ly_5_bc_op/"]
    Lys = [4, 5]
    Lxs = [12, 12]
    for ind, case in enumerate(cases):
        Lx = Lxs[ind]
        Ly = Lys[ind]
        print("#############")
        print(f"Lx={Lx}")
        print("#############")
        with open(main_dir + case + "psi_gutzwiller.pkl", 'rb') as f:
            psi = pickle.load(f)
        PermuteGutzwillerWavefunctionToDMRGOrder(psi, Lx, Ly)
        with open(main_dir + case + "psi_gutzwiller_permuted.pkl", 'wb') as f:
            pickle.dump(psi, f)


if __name__ == "__main__":
    # PlotSquareLatticeStructureFactor(Lx=3, Ly=3)
    # TestTriangularLattice()
    # TestSquareLattice(6, 6, J2s=[0.0, 0.9])

    #TriangularPiFluxAnsatz(spinfull=True, Lx=4, Ly=4,
    #                       chi_max_temfpy=100, bc_MPS="finite")

    # TriangularPiFluxGutzwiller(3, Lx=3, chi_max=100)

    #unit_cell_spin_lat = [[0.0, 0.0], [1.0, 0.0]]
    #basis = [[2.0, 0.0], [0.5, sqrt(3) / 2.]]
    #lat = BuildTriangularLatticeAlignedWithX(4, 4, SpinHalfSite(), "finite",
    #                                         unit_cell=unit_cell_spin_lat, basis=basis)
    #fig,ax = plt.subplots()
    #PlotLattice(lat, ax)
    #plt.show()

    # TestCorrelationsWithNontrivialUnitCell("stripe")

    # TriangularPiFluxGutzwiller(4)
    #PlotRealSpaceCorrelations(code_dir + "TriangularPiFluxGutzwiller/Lx_3_Ly_6_chi_6000/")
    #PlotRealSpaceCorrelations(code_dir +
    #                          "NewTenpyTriangularLatticeResults/Lx_6_Ly_6_bc_op/mps_finite_flux_0.0_init_Random_conserve_1_J2_0.125/")

    # plt.show()
    # ComputeCorrelationsFromMPSFile(code_dir + "TriangularLatticeResults/FromCluster/", 12, 5, ("open", "periodic"),
    #                              "finite", 0.0, "Random", 1, 0.125)

    #ComputeCorrelationsFromMPSFile(code_dir + "TriangularLatticeResults/FromCluster/", 1, 6, ("periodic", "periodic"),
    #                               "infinite", 0.0, "Random", 1, 0.125, Lx_for_infinite_bc_MPS=12)

    # calculateGutzwillerEnergyTriangularJ1J2(6, 6, reorder_lattice=False)

    #Lx = 6
    #Ly = 6
    #ComputeCorrelationsFromMPSFile(code_dir + "TriangularPiFluxGutzwiller/", 2*Lx, Ly, ("open", "periodic"),
    #                               "finite", psi_fname="psi_gutzwiller.pkl", sort_charge=True,
    #                               psi_dir=f"Lx_{Lx}_Ly_{Ly}_chi_6000/", from_dmrg=False, from_corr_file=True)

    #TriangularJ1J2DMRG(3, 3, ("open", "periodic"), "finite",
    #                   J2=0.125, conserve=True, initial_state="Random")
    # PermuteGutzwillerResults()
    # GutzwillerDMRGComparisons()
    # DMRGCorrelations()
    GutzwillerDMRGComparisons()
    # PermuteGutzwillerResults()
