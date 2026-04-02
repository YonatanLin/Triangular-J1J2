from temfpy import slater
import tenpy
import numpy as np
from numpy import log, sin, cos, sqrt, pi
import matplotlib.pyplot as plt
from tenpy import NearestNeighborModel, CouplingModel, SpinHalfSite, TwoSiteDMRGEngine, MPS, CouplingMPOModel, SpinModel
import tenpy.linalg.np_conserved as npc
from tenpy.models import lattice
from tenpy.algorithms import dmrg
from numpy.linalg import norm, eigh
from tenpy.tools.misc import setup_logging

setup_logging(to_stdout="INFO")

fontsize=18
rc_params = {
    "font.family": "serif",
    "figure.dpi": 200,
    'text.usetex': True,
    #"axes.labelsize": 50,
    #"axes.titlesize": 50,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "figure.titlesize": fontsize,
    "legend.loc": "upper left",
}
plt.rcParams.update(rc_params)

def add_and_track_coupling(model, strength, u1, op1, u2, op2, dx, couplings_list):
    model.add_coupling(strength, u1, op1, u2, op2, dx)
    couplings_list.append((u1, u2, dx))

def getSpinSpinCorrelations(psi):
    pm_corr = psi.correlation_function("Sp", "Sm")
    mp_corr = psi.correlation_function("Sm", "Sp")
    zz_corr = psi.correlation_function("Sz", "Sz")
    spin_corr_x = 0.5 * (pm_corr + mp_corr) + zz_corr
    return spin_corr_x

def ComputeMomentumSpaceStructureFactor(corr_x, lat, Lx, Ly):
    kx = ky = np.linspace(-1.5*np.pi, 1.5*np.pi, 100)
    Kx, Ky = np.meshgrid(kx, ky)
    lat_basis = lat.basis
    center_site_coordinates = [Lx // 2, Ly // 2, 0]
    center_site_mps_index = lat.lat2mps_idx(center_site_coordinates)
    print("center site index for calculating correlations: ", center_site_mps_index)

    basis_vectors = np.asarray(lat_basis[:, :2], dtype=float)
    center_site_loc = np.dot(center_site_coordinates[:2], basis_vectors)
    print("center site index for calculating correlations: ", center_site_mps_index)
    spin_corr_with_center = corr_x[center_site_mps_index, :]
    site_coordinates = np.asarray(lat.order[:, :2], dtype=float)
    site_locations = site_coordinates @ basis_vectors
    r_from_center = site_locations - center_site_loc
    phases = np.exp(1j * (Kx[..., np.newaxis] * r_from_center[:, 0] +
                          Ky[..., np.newaxis] * r_from_center[:, 1]))
    spin_corr_k = np.sum(spin_corr_with_center[np.newaxis, np.newaxis, :] * phases, axis=2)

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

    spin_corr_x = getSpinSpinCorrelations(psi)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, square_lat, Lx, Ly)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        np.real(spin_corr_k),
        origin='lower',
        extent=[Kx.min(), Kx.max(), Ky.min(), Ky.max()],
        cmap='magma',
        aspect='auto',
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$S(\mathbf{k})$")
    ax.set_title(f"Spin structure factor on a {Lx}x{Ly} square lattice")
    ax.set_xlabel(r"$k_x / \pi$")
    ax.set_ylabel(r"$k_y / \pi$")
    ax.axhline(0.0, color='white', linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color='white', linewidth=0.6, alpha=0.5)

    square_lat.plot_brillouin_zone(ax)

    fig.tight_layout()
    plt.show()


def Generate120DegOrderedState():
    site = SpinHalfSite(conserve=None)
    Lx = 9
    Ly = 9
    triangular_lat = lattice.Triangular(Lx=Lx, Ly=Ly, site=site, bc=['periodic', 'open'])
    basis = triangular_lat.basis
    assert(basis[1][0] == 0.0 and basis[1][1] == 1.0)

    psi = MPS.from_product_state(triangular_lat.mps_sites(), ["up"] * triangular_lat.N_sites)

    rot_120_angle = 2 * pi / 3
    debug = abs(rot_120_angle - pi) < 1e-15
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    I2 = np.eye(2)
    rotation_op_120 = cos(rot_120_angle / 2.) * I2 - 1j * sin(rot_120_angle / 2.) * pauli_x
    rotation_op_240 = cos(rot_120_angle) * I2 - 1j * sin(rot_120_angle) * pauli_x
    site.add_op("rot_once", rotation_op_120)
    site.add_op("rot_twice", rotation_op_240)

    for i in range(triangular_lat.N_sites):
        lat_ind = triangular_lat.mps2lat_idx(i)
        y_cor = lat_ind[0] * basis[0][1] + lat_ind[1] * basis[1][1]
        sublattice_ind = int((2 * y_cor) % 3)
        if sublattice_ind == 1:
            psi.apply_local_op(i, "rot_once")
        elif sublattice_ind == 2:
            psi.apply_local_op(i, "rot_twice")

    psi.canonical_form()
    magz = psi.expectation_value("Sz")
    fig_lat, ax_lat = plt.subplots()
    triangular_lat.plot_order(ax_lat)
    triangular_lat.plot_coupling(ax_lat)
    triangular_lat.plot_sites(ax_lat)

    if debug and Lx == 4 and Ly == 4:
        magz_expected = 0.5 * np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1])
        assert ((magz - magz_expected) == 0.0).all()

    spin_corr_x = getSpinSpinCorrelations(psi)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lat, Lx, Ly)
    fig_corr, ax_corr = plt.subplots()
    kx_min = np.min(Kx[0,:])
    kx_max = np.max(Kx[0,:])
    ky_min = np.min(Ky[:,0])
    ky_max = np.max(Ky[:,0])

    ax_corr.imshow(np.abs(spin_corr_k), cmap='RdBu', extent=[kx_min, kx_max, ky_min, ky_max])
    ax_corr.set_xlabel(r"$ k_x [ \pi ]$")
    ax_corr.set_ylabel(r"$ k_y [ \pi ]$")

    triangular_lat.plot_brillouin_zone(ax_corr)
    plt.show()


def TestTriangularLattice():
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    site = SpinHalfSite(conserve='Sz')
    # site = SpinHalfSite()
    Lx = 6
    Ly = 6
    triangular_lat = lattice.Triangular(Lx=Lx, Ly=Ly, site=site, bc=['periodic', 'open'])
    triangular_lat_basis = triangular_lat.basis
    print("lattice basis: ", triangular_lat_basis)
    print("lattice ordering: ", triangular_lat.order)
    print("lattice site (0,1) mps index: ", triangular_lat.lat2mps_idx([0,1,0]))
    center_site_mps_index = triangular_lat.lat2mps_idx([Lx//2,Ly//2,0])
    print("center site mps index: ", center_site_mps_index)
    J1 = 1.0
    # J2 = 1. / 8.
    J2 = 0.0

    J1J2_model = SpinModel({"lattice":triangular_lat, "Jx":J1, "Jy":J1, "Jz":J1})
    J1J2_model.manually_call_init_H = True
    add_nn_explicitly = False
    nn_couplings_list = []
    if add_nn_explicitly:
        for basis_vec in ([1, 0], [0, 1]):
            add_and_track_coupling(J1J2_model,0.5 * J1, 0, "Sp", 0, "Sm", basis_vec,
                                   nn_couplings_list)
            add_and_track_coupling(J1J2_model, 0.5 * J1, 0, "Sm", 0, "Sp", basis_vec,
                                   nn_couplings_list)
            add_and_track_coupling(J1J2_model, J1, 0, "Sz", 0, "Sz", basis_vec,
                                   nn_couplings_list)

    nnn_couplings_list = []
    if abs(J2) > 0.0:
        for basis_vec in ([1,1], [-1,-1], [-1,2], [1,-2], [-2,1], [2,-1]):
            add_and_track_coupling(J1J2_model,0.5 * J2, 0, "Sp", 0, "Sm", basis_vec,
                                   nnn_couplings_list)
            add_and_track_coupling(J1J2_model, 0.5 * J2, 0, "Sm", 0, "Sp", basis_vec,
                                   nnn_couplings_list)
            add_and_track_coupling(J1J2_model, J2, 0, "Sz", 0, "Sz", basis_vec,
                                   nnn_couplings_list)

    J1J2_model.init_H_from_terms()

    print_couplings = False
    if print_couplings:
        print("couplings: ")
        couplings_list = J1J2_model.all_coupling_terms().to_TermList()
        fancy_print = False
        if fancy_print:
            for i in range(Lx*Ly):
                for j in range(i, Lx*Ly):
                    for coupling in couplings_list:
                        if(coupling[0][0][1] == i and coupling[0][1][1] == j):
                            print(coupling)
        else:
            print(couplings_list)

    do_dmrg = True
    if do_dmrg:
        psi = MPS.from_product_state(triangular_lat.mps_sites(), ["up" for i in range(triangular_lat.N_sites//2)] +
                                     ["down" for i in range(triangular_lat.N_sites - triangular_lat.N_sites//2)],
                                    bc=triangular_lat.bc_MPS, unit_cell_width=triangular_lat.mps_unit_cell_width)

        dmrg_params = {'mixer': True, 'max_E_err': 1.0e-10, 'trunc_params': {'chi_max': 500, 'svd_min': 1.0e-10},
                       'combine': True, 'chi_list': {0: 20, 10: 50, 20: 500},  'min_sweeps': 40, 'max_sweeps': 200,
                       'N_sweeps_check': 5}

        info = dmrg.run(psi, J1J2_model, dmrg_params)
        E = info['E']
        print(f'E = {E:.13f}')
        print('final bond dimensions: ', psi.chi)
        mag_z = np.sum(psi.expectation_value('Sz'))
        print(f'magnetization in Z = {mag_z:.5f}')
        stats = info['sweep_statistics']
        energies = stats['E']
        sweeps = stats['sweep']

        pm_corr = psi.correlation_function("Sp", "Sm")
        mp_corr = psi.correlation_function("Sm", "Sp")
        zz_corr = psi.correlation_function("Sz", "Sz")
        spin_corr_x = 0.5 * (pm_corr + mp_corr) + zz_corr

        spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lat, Lx, Ly)
        plt.imshow(spin_corr_k, cmap='RdBu')
        plt.show()
        plt.plot(sweeps, energies, "o")
        plt.show()

    plot = True
    if plot:
        if add_nn_explicitly:
            triangular_lat.plot_coupling(ax, coupling=nn_couplings_list, linewidth=1.0)
        else:
            triangular_lat.plot_coupling(ax, linewidth=1.0)
        triangular_lat.plot_coupling(ax, coupling=nnn_couplings_list, linewidth=0.5,
                          color="green")
        triangular_lat.plot_order(ax)
        # triangular_lat.plot_coupling(ax)
        triangular_lat.plot_sites(ax)
        triangular_lat.plot_basis(ax, origin=-0.5*(triangular_lat.basis[0] + triangular_lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()


if __name__ == "__main__":
    PlotSquareLatticeStructureFactor(Lx=3, Ly=3)
    Generate120DegOrderedState()
    # TestTriangularLattice()
