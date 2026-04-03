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
    pm_corr = psi.correlation_function("Sp", "Sm")
    mp_corr = psi.correlation_function("Sm", "Sp")
    zz_corr = psi.correlation_function("Sz", "Sz")
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


def Generate120DegOrderedState(lat=None, Lx=None, Ly=None):
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
    if debug:
        fig_lat, ax_lat = plt.subplots()
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        spin_corr_x = GetSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat, Lx, Ly)
        fig_corr, ax_corr = plt.subplots()
        kx_min = np.min(Kx[0,:])
        kx_max = np.max(Kx[0,:])
        ky_min = np.min(Ky[:,0])
        ky_max = np.max(Ky[:,0])

        ax_corr.imshow(np.abs(spin_corr_k), cmap='RdBu', extent=[kx_min, kx_max, ky_min, ky_max])
        ax_corr.set_xlabel(r"$ k_x [ \pi ]$")
        ax_corr.set_ylabel(r"$ k_y [ \pi ]$")

        lat.plot_brillouin_zone(ax_corr)
        plt.show()

    return psi


def RunDMRG(model, psi_init, plot_convergence=False, print_final_results=False,
            expected_energy=None):
    chi_max = 1600
    dmrg_params = {'mixer': True, 'max_E_err': 1.0e-10, 'trunc_params': {'chi_max': chi_max, 'svd_min': 1.0e-10},
                   'combine': True, 'chi_list': {0: 20, 5: 50, 10: chi_max}, 'min_sweeps': 25, 'max_sweeps': 50,
                   'N_sweeps_check': 1}

    info = dmrg.run(psi_init, model, dmrg_params)
    E = info['E']
    stats = info['sweep_statistics']
    energies = stats['E']
    sweeps = stats['sweep']
    if print_final_results:
        print(f'E = {E:.13f}')
        print('final bond dimensions: ', psi_init.chi)
        #mag_z = np.sum(psi_init.expectation_value('Sz'))
        #print(f'magnetization in Z = {mag_z:.5f}')

    if plot_convergence:
        fig,ax = plt.subplots()
        ax.plot(sweeps, energies, "o")
        ax.set_title("DMRG Sweeps Energies")
        ax.set_xlabel("sweep")
        ax.set_ylabel("E")
        if expected_energy is not None:
            ax.axhline(expected_energy, color="red", linewidth=0.6, alpha=0.5, linestyle="dashed")
        plt.show()
    return E, sweeps, energies, info


def TestSquareLattice(Lx=5, Ly=5, bc=('open', 'open'), J2s=[0.0]):
    assert(Lx == Ly)
    L = Lx
    for J2 in J2s:
        site = SpinHalfSite(conserve='Sz')
        square_lat = lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=list(bc))
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

        fig_lat, ax_lat = plt.subplots()
        PlotLattice(square_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)

        psi = MPS.from_product_state(
            square_lat.mps_sites(),
            ["up"] * (square_lat.N_sites // 2) + ["down"] * (square_lat.N_sites // 2),
            bc=square_lat.bc_MPS,
            unit_cell_width=square_lat.mps_unit_cell_width,
        )
        psi.canonical_form()

        initial_energy = J1J2_model.H_MPO.expectation_value(psi)
        initial_energy_per_site = initial_energy / square_lat.N_sites
        #if J2 == 0.0:
        #    assert(initial_energy_per_site == 0.5 * (1. - 1. / L))
        print("Initial energy per site: ", initial_energy / square_lat.N_sites)
        E, sweeps, energies, _ = RunDMRG(
            J1J2_model,
            psi,
            plot_convergence=False,
            print_final_results=True,
        )
        energy_per_site = E / square_lat.N_sites
        print(f"Energy per site = {energy_per_site:.13f}")

        spin_corr = GetSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(
            spin_corr,
            square_lat,
            Lx,
            Ly,
            assert_realness=False,
        )

        fig_energy, ax_energy = plt.subplots(figsize=(6, 4))
        energies = [initial_energy] + energies
        ax_energy.plot([-1] + sweeps, energies, "o-", linewidth=1.0, markersize=4)
        ax_energy.set_title(f"Heisenberg square lattice DMRG convergence ({Lx}x{Ly})")
        ax_energy.set_xlabel("sweep")
        ax_energy.set_ylabel("E")
        fig_energy.tight_layout()
        fig_energy.savefig("SquareLatticeJ1J2/energy_convergence_J2_" + str(J2) + ".png", bbox_inches='tight')

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
    lattice.Triangular()
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
            AddAndTrackCoupling(self, 1.0, s2, "Cd", s2, "C", [-1, 0], pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, -1.0, s1, "Cd", s1, "C", [-1, 0], neg_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, 1.0, s2, "Cd", s1, "C", [0, 0], pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, -1.0, s2, "Cd", s1, "C", [1, 0], neg_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, 1.0, s1, "Cd", s2, "C", np.array([-1, 0]) - np.array(rec_long_side_coors),
                                pos_hoppings_list, plus_hc=plus_hc)
            AddAndTrackCoupling(self, 1.0, s1, "Cd", s2, "C", (-1) * np.array(rec_long_side_coors),
                                pos_hoppings_list, plus_hc=plus_hc)

        self.neg_hoppings_list = neg_hoppings_list
        self.pos_hoppings_list = pos_hoppings_list


def TriangularPiFluxAnsatz(Lx=3, Ly=2, spinfull=True):
    site = FermionSite(conserve='N')

    lat_basis = [[0.5, sqrt(3) / 2.], [2.0, 0]]
    if spinfull:
        unitcell_pos = [[-0.1, 0.0],  [0.1, 0.0], [0.9, 0.0], [1.1, 0.0]]
    else:
        unitcell_pos = [[0.0, 0.0], [1.0, 0.0]]

    rec_long_side_coors = [-1, 1]
    #nearest_neighbors = [(1, 1, [1, 0]), (0, 0, [1, 0]), (0,1,[0,0]), (0,1,[-1,0]),
    #                      (1,0,[0,1]), (1,0,[-1,1])]
    nearest_neighbors = []

    triangular_lat = lattice.Lattice([Lx, Ly], [site] * len(unitcell_pos), basis=lat_basis,
                                     positions=unitcell_pos, bc=['periodic', 'open'],
                                     pairs={'nearest_neighbors': nearest_neighbors},
                                     order=('standard', (False, True, False), (1,0,2)))

    pi_flux_model = FermionicPiFluxModel({"lattice": triangular_lat, "rec_long_side_coors":rec_long_side_coors,
                                          "spinfull":spinfull})
    # PrintCouplings(pi_flux_model, Lx, Ly)
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

    H = CreateHamiltonianMatrixFromCouplingsList(couplings_list, N_sites)
    if Lx > 2:
        assert(np.max(np.abs(H)) < 1. + 1e-15)
    C, _ = slater.correlation_matrix(H)
    trunc_par = {"chi_max": 2000, "svd_min": 1e-6, "degeneracy_tol": 1e-12}
    psi_from_slater = slater.C_to_MPS(C, trunc_par)

    mps_slater_corr = psi_from_slater.correlation_function("Cd", "C")
    
    psi_dmrg = MPS.from_product_state(triangular_lat.mps_sites(),
                                      ["full"] * (N_sites // 2) + ["empty"] * (N_sites // 2),
                                      bc=triangular_lat.bc_MPS)
    E_slater_mps = pi_flux_model.H_MPO.expectation_value(psi_from_slater)
    RunDMRG(pi_flux_model, psi_dmrg, plot_convergence=True, print_final_results=True, expected_energy=E_slater_mps)
    dmrg_corr = psi_dmrg.correlation_function("Cd", "C")

    print("Energy for mps-slater:", E_slater_mps)
    print("Energy from dmrg:", pi_flux_model.H_MPO.expectation_value(psi_dmrg))

    X,Y = np.meshgrid(np.arange(0,N_sites),np.arange(0,N_sites))
    fig_slater_corr, ax_slater_corr = plt.subplots()
    fig_mps_slater_corr, ax_mps_slater_corr = plt.subplots()
    fig_dmrg_corr, ax_dmrg_corr = plt.subplots()
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
    print("correlations max distance between slater-mps and exact slater: ", np.max(np.abs(mps_slater_corr - C)))
    print("correlations max distance between slater-mps and dmrg: ", np.max(np.abs(mps_slater_corr - dmrg_corr)))
    print("slater norm: ", psi_from_slater.norm)
    print("dmrg norm: ", psi_dmrg.norm)



if __name__ == "__main__":
    #PlotSquareLatticeStructureFactor(Lx=3, Ly=3)
    #Generate120DegOrderedState()
    # TestTriangularLattice()
    # TestSquareLattice(6, 6, J2s=[0.0, 0.9])

    TriangularPiFluxAnsatz(spinfull=True, Lx=3, Ly=2)

