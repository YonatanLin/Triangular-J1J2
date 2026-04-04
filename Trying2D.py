from temfpy import slater
import tenpy
import numpy as np
from numpy import log, sin, cos, sqrt, pi
import matplotlib.pyplot as plt
from tenpy import NearestNeighborModel, CouplingModel, SpinHalfSite, TwoSiteDMRGEngine, MPS, CouplingMPOModel, \
    SpinModel, FermionSite, FermionModel, Lattice, SpinHalfFermionSite
import tenpy.linalg.np_conserved as npc
from tenpy.models import lattice
from tenpy.algorithms import dmrg
from numpy.linalg import norm, eigh
from tenpy.tools.misc import setup_logging
import pickle

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



def GetSpinSpinCorrelations(psi):
    sites1 = None
    sites2 = None
    if psi.bc == "infinite":
        L = psi.L
        unit_cell_width = psi.unit_cell_width
        sites1 = np.arange(L//2,L//2+unit_cell_width)
        sites2 = np.arange(0,L)

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
                      color="green", linestyle=nnn_line_style)
    if plot_order:
        lat.plot_order(ax)
    lat.plot_sites(ax)
    lat.plot_basis(ax, origin=-0.5 * (lat.basis[0] + lat.basis[1]))
    ax.set_aspect('equal')
    ax.set_xlim(-1)
    ax.set_ylim(-1)

def ImshowMatrix(ax, fig, X, Y, spin_corr_k, xlabel = r"$k_x / \pi$",
                                      ylabel=r"$k_y / \pi$", title=None):
    image = ax.imshow(
        np.real(spin_corr_k),
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='RdBu',
        aspect='auto',
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$S(\mathbf{k})$")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color='white', linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color='white', linewidth=0.6, alpha=0.5)

def ComputeMomentumSpaceStructureFactor(corr_x, lat, Lx, Ly, assert_realness=True):
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

    spin_corr_x = GetSpinSpinCorrelations(psi)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, square_lat, Lx, Ly)

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
    assert(basis[1][0] == 0.0 and basis[1][1] == 1.0)

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
        y_cor = lat_ind[0] * basis[0][1] + lat_ind[1] * basis[1][1]
        sublattice_ind = int((2 * y_cor) % 3)
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
        spin_corr_x = GetSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat, Lx, Ly)
        fig_corr, ax_corr = plt.subplots()
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k)
        lat.plot_brillouin_zone(ax_corr)
        plt.show()

    return psi


def RunDMRG(model, psi_init, plot_convergence=False, print_final_results=False,
            expected_energy=None, results_dir="", energies_fig_title=None):
    chi_max = 1600
    # chi_max = 300
    dmrg_params = {'mixer': True, 'max_E_err': 1.0e-10, 'trunc_params': {'chi_max': chi_max, 'svd_min': 1.0e-10},
                   'combine': True, 'chi_list': {0: 20, 5: 50, 10: chi_max}, 'min_sweeps': 15, 'max_sweeps': 50,
                   'N_sweeps_check': 1}
    E_initial = model.H_MPO.expectation_value(psi_init)
    info = dmrg.run(psi_init, model, dmrg_params)
    E_final = info['E']
    stats = info['sweep_statistics']
    energies = stats['E']
    sweeps = stats['sweep']
    if print_final_results:
        print(f'E = {E_final:.13f}')
        print('final bond dimensions: ', psi_init.chi)
        #mag_z = np.sum(psi_init.expectation_value('Sz'))
        #print(f'magnetization in Z = {mag_z:.5f}')

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

        results_folder = "SquareLatticeJ1J2/"
        fig_lat, ax_lat = plt.subplots()
        PlotLattice(square_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)
        fig_lat.savefig(results_folder + "lattice.png", bbox_inches='tight')

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
        E_initial, E_gs, sweeps, energies, info = RunDMRG(J1J2_model, psi, plot_convergence=True, print_final_results=True,
                                                       results_dir=results_folder, energies_fig_title=energies_fig_title)
        energy_per_site = E_gs

        with open("Energy_J2_"+str(J2)+".txt", "w") as f:
            f.write(f"Energy per site = {energy_per_site:.13f}")

        if bc_MPS == "finite":
            energy_per_site /= square_lat.N_sites
        print(f"Energy per site = {energy_per_site:.13f}")

        with open(results_folder + 'psi_gs_J2_'+str(J2)+".pkl", 'wb') as f:
            pickle.dump(psi, f)

        spin_corr = GetSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr, square_lat, Lx, Ly,
                                                                  assert_realness=False)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k, title=title)
        square_lat.plot_brillouin_zone(ax_corr)
        fig_corr.tight_layout()

        fig_corr.savefig("SquareLatticeJ1J2/spin_correlations_J2_"+str(J2)+".png", bbox_inches='tight')

        plt.show()
    # return energy_per_site, psi, square_lat


def TestTriangularLattice():
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    conserve = False
    if conserve:
        site = SpinHalfSite(conserve='Sz')
    else:
        site = SpinHalfSite(conserve=None)
    Lx = 5
    Ly = 5
    triangular_lat = lattice.Triangular(Lx=Lx, Ly=Ly, site=site, bc=['open', 'open'])
    triangular_lat_basis = triangular_lat.basis
    center_site_mps_index = triangular_lat.lat2mps_idx([Lx // 2, Ly // 2, 0])
    print("lattice basis: ", triangular_lat_basis)
    print("lattice ordering: ", triangular_lat.order)
    print("lattice site (0,1) mps index: ", triangular_lat.lat2mps_idx([0,1,0]))
    print("center site mps index: ", center_site_mps_index)
    J1 = 1.0
    J2 = 0.0

    J1J2_model = SpinModel({"lattice":triangular_lat, "Jx":J1, "Jy":J1, "Jz":J1})
    J1J2_model.manually_call_init_H = True

    nnn_couplings_list = []
    if abs(J2) > 0.0:
        for basis_vec in ([1,1], [-1,-1], [-1,2], [1,-2], [-2,1], [2,-1]):
            AddAndTrackCoupling(J1J2_model,0.5 * J2, 0, "Sp", 0, "Sm", basis_vec,
                                   nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sm", 0, "Sp", basis_vec,
                                   nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, J2, 0, "Sz", 0, "Sz", basis_vec,
                                   nnn_couplings_list)
    J1J2_model.init_H_from_terms()

    print_couplings = False
    if print_couplings:
        PrintCouplings(J1J2_model, Lx, Ly)

    do_dmrg = True
    start_with_product_state = False
    if do_dmrg:
        if start_with_product_state:
            magz = int(np.ceil(triangular_lat.N_sites / 2))
            psi = MPS.from_product_state(triangular_lat.mps_sites(), ["up" for i in range(triangular_lat.N_sites // 2 + magz)] +
                                         ["down" for i in range(triangular_lat.N_sites - triangular_lat.N_sites // 2 - magz)],
                                         bc=triangular_lat.bc_MPS, unit_cell_width=triangular_lat.mps_unit_cell_width)
        else:
            psi = Generate120DegOrderedState(lat=triangular_lat, Lx=Lx, Ly=Ly)
        RunDMRG(J1J2_model, psi, True, True)
        spin_corr_x = GetSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lat, Lx, Ly, assert_realness=False)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k, title=title)


    plot = True
    if plot:
        PlotLattice(triangular_lat, ax, additional_couplings_to_plot=nnn_couplings_list)


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
        if spinfull:
            unitcell_hopping_pairs = [(0,2), (1,3)]
        else:
            unitcell_hopping_pairs = [(0, 1)]
        for pair in unitcell_hopping_pairs:
            s1 = pair[0]
            s2 = pair[1]
            AddAndTrackCoupling(self, 1.0, s2, "Cd", s2, "C", [0,-1], pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, -1.0, s1, "Cd", s1, "C", [0,-1], neg_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, 1.0, s2, "Cd", s1, "C", [0, 0], pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, -1.0, s2, "Cd", s1, "C", [0,1], neg_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, 1.0, s1, "Cd", s2, "C", np.array([0,-1]) - np.array(rec_long_side_coors),
                                pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, 1.0, s1, "Cd", s2, "C", (-1) * np.array(rec_long_side_coors),
                                pos_hoppings_list, plus_hc=plus_hc)

        self.neg_hoppings_list = neg_hoppings_list
        self.pos_hoppings_list = pos_hoppings_list


def TriangularPiFluxAnsatz(Lx=2, Ly=3, spinfull=True, bc_MPS="infinite"):
    site = FermionSite(conserve='N')
    lat_basis = [[2.0, 0], [0.5, sqrt(3) / 2.]]
    if spinfull:
        unitcell_pos = [[-0.1, 0.0],  [0.1, 0.0], [0.9, 0.0], [1.1, 0.0]]
    else:
        unitcell_pos = [[0.0, 0.0], [1.0, 0.0]]

    rec_long_side_coors = [1, -1]
    #nearest_neighbors = [(1, 1, [1, 0]), (0, 0, [1, 0]), (0,1,[0,0]), (0,1,[-1,0]),
    #                      (1,0,[0,1]), (1,0,[-1,1])]
    nearest_neighbors = []
    bc = ['periodic', 'open']
    if bc_MPS == "infinite":
        bc[1] = 'periodic'
    pairs = {'nearest_neighbors': nearest_neighbors}
    order = ('standard', (True, False, False), (0,1,2))
    triangular_lat = lattice.Lattice([Lx, Ly], [site] * len(unitcell_pos), basis=lat_basis,
                                     positions=unitcell_pos, bc=bc, pairs=pairs, order=order, bc_MPS=bc_MPS)

    pi_flux_model = FermionicPiFluxModel({"lattice": triangular_lat, "rec_long_side_coors":rec_long_side_coors,
                                          "spinfull":spinfull})
    middle_site_mps_ind = triangular_lat.lat2mps_idx([Lx // 2 - 1, Ly - 1, 3])
    print("middle site: ", middle_site_mps_ind)
    plot_lattice = True
    if plot_lattice:
        fig_lat, ax_lat = plt.subplots()
        plot_order = True
        PlotLattice(triangular_lat, ax_lat, pi_flux_model.pos_hoppings_list, plot_nn_couplings=False, plot_order=plot_order)
        PlotLattice(triangular_lat, ax_lat, pi_flux_model.neg_hoppings_list, plot_nn_couplings=False,
                    nnn_line_style="--", plot_order=plot_order)
        fig_lat.savefig("PiFluxAnsatzResults/lattice.png", bbox_inches='tight')
        plt.show()

    N_sites = triangular_lat.N_sites

    pi_flux_model.init_H_from_terms()
    couplings_list = pi_flux_model.all_coupling_terms().to_TermList()

    chi_max = 5000
    slater_trunc_par = {"chi_max": chi_max, "svd_min": 1e-7, "degeneracy_tol": 1e-12}

    mps_unitcell = len(unitcell_pos) * Ly
    if bc_MPS == "finite":
        H = CreateHamiltonianMatrixFromCouplingsList(couplings_list, N_sites)
        if Lx > 2:
            assert(np.max(np.abs(H)) < 1. + 1e-15)
        C, _ = slater.correlation_matrix(H)
        psi_from_slater = slater.C_to_MPS(C, slater_trunc_par)

    else:
        Lx_short, Ly_short = 100, Ly
        Lx_long, Ly_long = 101, Ly
        triangular_lat_short = lattice.Lattice([Lx_short, Ly_short], [site] * len(unitcell_pos), basis=lat_basis,
                                         positions=unitcell_pos, bc=bc, pairs=pairs, order=order, bc_MPS="finite")
        pi_flux_model_short = FermionicPiFluxModel({"lattice": triangular_lat_short,
                                                    "rec_long_side_coors":rec_long_side_coors,
                                                    "spinfull":spinfull})

        triangular_lat_long = lattice.Lattice([Lx_long, Ly_long], [site] * len(unitcell_pos), basis=lat_basis,
                                               positions=unitcell_pos, bc=bc, pairs=pairs, order=order, bc_MPS="finite")
        pi_flux_model_long = FermionicPiFluxModel({"lattice": triangular_lat_long,
                                                   "rec_long_side_coors": rec_long_side_coors,
                                                   "spinfull": spinfull})
        couplings_list_short = pi_flux_model_short.all_coupling_terms().to_TermList()
        couplings_list_long = pi_flux_model_long.all_coupling_terms().to_TermList()
        H_short = CreateHamiltonianMatrixFromCouplingsList(couplings_list_short, triangular_lat_short.N_sites)
        H_long = CreateHamiltonianMatrixFromCouplingsList(couplings_list_long, triangular_lat_long.N_sites)

        C_short, _ = slater.correlation_matrix(H_short)
        C_long, _ = slater.correlation_matrix(H_long)

        middle_site_mps_ind_short = triangular_lat_short.lat2mps_idx([Lx_short // 2, 0, 0])

        psi_from_slater, error = slater.C_to_iMPS(C_short, C_long, slater_trunc_par,
                                                  sites_per_cell=mps_unitcell,
                                                  cut= middle_site_mps_ind_short)
        C = []

    sites1 = None
    sites2 = None
    if bc_MPS == "infinite":
        sites1 = np.arange(0, 10 * mps_unitcell)
        sites2 = np.arange(0, 10 * mps_unitcell)

    mps_slater_corr = psi_from_slater.correlation_function("Cd", "C", sites1=sites1, sites2=sites2)
    E_slater_mps = pi_flux_model.H_MPO.expectation_value(psi_from_slater)
    print("Energy for mps-slater:", E_slater_mps)

    psi_dmrg = MPS.from_product_state(triangular_lat.mps_sites(),
                                      ["full"] * (N_sites // 2) + ["empty"] * (N_sites // 2),
                                      bc=triangular_lat.bc_MPS)
    RunDMRG(pi_flux_model, psi_dmrg, plot_convergence=True, print_final_results=True, expected_energy=E_slater_mps)
    dmrg_corr = psi_dmrg.correlation_function("Cd", "C", sites1=sites1, sites2=sites2)

    print("Energy for mps-slater:", E_slater_mps)
    print("Energy from dmrg:", pi_flux_model.H_MPO.expectation_value(psi_dmrg))

    X,Y = np.meshgrid(np.arange(0,N_sites),np.arange(0,N_sites))
    fig_slater_corr, ax_slater_corr = plt.subplots()
    fig_mps_slater_corr, ax_mps_slater_corr = plt.subplots()
    fig_dmrg_corr, ax_dmrg_corr = plt.subplots()

    if bc == "finite":
        ImshowMatrix(ax_slater_corr, fig_slater_corr, X, Y, C, "i", "j")
    ImshowMatrix(ax_mps_slater_corr, fig_mps_slater_corr, X, Y, mps_slater_corr, "i", "j")
    ImshowMatrix(ax_dmrg_corr, fig_mps_slater_corr, X, Y, dmrg_corr, "i", "j")

    np.savetxt("PiFluxAnsatzResults/C_slater.csv", C)
    np.savetxt("PiFluxAnsatzResults/C_slater_mps.csv", mps_slater_corr)
    np.savetxt("PiFluxAnsatzResults/C_dmrg.csv", dmrg_corr)

    fig_slater_corr.savefig("PiFluxAnsatzResults/slater_exact_correlations.png", bbox_inches='tight')
    fig_mps_slater_corr.savefig("PiFluxAnsatzResults/slater_mps_correlations.png", bbox_inches='tight')
    fig_dmrg_corr.savefig("PiFluxAnsatzResults/dmrg_correlations.png", bbox_inches='tight')

    print("absolute value of overlap between slater and dmrg: ", np.abs(psi_dmrg.overlap(psi_from_slater)))
    print("correlations max distance between slater-mps and dmrg: ", np.max(np.abs(mps_slater_corr - dmrg_corr)))
    if bc == "finite":
        print("correlations max distance between slater-mps and exact slater: ", np.max(np.abs(mps_slater_corr - C)))

    print("slater norm: ", psi_from_slater.norm)
    print("dmrg norm: ", psi_dmrg.norm)

if __name__ == "__main__":
    #PlotSquareLatticeStructureFactor(Lx=3, Ly=3)
    # Generate120DegOrderedState(plot=True)
    # TestTriangularLattice()
    # TestSquareLattice(6, 6, J2s=[0.0, 0.9])
    # TestSquareLattice(2, 3, J2s=[0.0], bc=("periodic", "periodic"), bc_MPS="infinite")
    TriangularPiFluxAnsatz(spinfull=True, Lx=1, Ly=3)

