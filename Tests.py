from WaveFunctionProperties import (plot_scalar_spin_chirality, compute_structure_factor_grid,
                                    plot_structure_factor, structure_factor, ComputeMomentumSpaceStructureFactor,
                                    CalculateSpinSpinCorrelations, CalculateDimerDimerCorrelations)
from numpy import sin, cos, sqrt, pi
import numpy as np
from tenpy.networks.site import FermionSite, SpinHalfSite
from Main import (BuildTriangularLattice, Generate120DegOrderedState, GenerateStripeOrderedState,
                  CreateHamiltonianMatrixFromCouplingsList, PrintCouplings, PlotLattice,
                  CalculateExactCMatrixForPiFlux, model_type_dirac, ImshowMatrix)
from tenpy.networks.mps import MPS
import matplotlib.pyplot as plt
from TryingTemfpy import local


def TestSquareLattice(Lx=5, Ly=5, bc=('open', 'open'), J2s=[0.0],
                      bc_MPS="finite"):
    import tenpy
    from tenpy.models.spins import SpinModel
    from Main import AddAndTrackCoupling, RunDMRG
    import pickle

    for J2 in J2s:
        site = SpinHalfSite(conserve='Sz')
        square_lat = tenpy.models.lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=list(bc), bc_MPS=bc_MPS)
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
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
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
        ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr, square_lat,
                                                                  assert_realness=False)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        title = f"Spin structure factor"
        ax_corr.set_title(title)
        plot_structure_factor(ks, spin_corr_k, square_lat, ax_corr)
        square_lat.plot_brillouin_zone(ax_corr)
        fig_corr.tight_layout()

        fig_corr.savefig("SquareLatticeJ1J2/spin_correlations_J2_"+str(J2)+".png", bbox_inches='tight')

        if local:
            plt.show()
    # return energy_per_site, psi, square_lat


def TestDimerDimerCorrelations():
    site = SpinHalfSite(conserve=None)
    lat = BuildTriangularLattice(6, 6, site, "finite", bc=("open", "open"), geometry="YC")
    psi = MPS.from_product_state(
        lat.mps_sites(),
        ["up"] * lat.N_sites,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )

    dimer_corr = CalculateDimerDimerCorrelations(psi, lat)
    expected_dimer_corr = np.zeros((lat.N_sites, lat.N_sites))
    assert(dimer_corr.shape == expected_dimer_corr.shape)
    assert(np.max(np.abs(dimer_corr - expected_dimer_corr)) < 1e-13)

    partial_corr = CalculateDimerDimerCorrelations(psi, lat, sites1=[0, 3], sites2=[0, 3])
    assert(partial_corr.shape == (2, 2))
    assert(np.max(np.abs(partial_corr)) < 1e-13)
    print("success")


def TestCorrelationsWithNontrivialUnitCell(Lx, Ly, state="120", geometry="YC"):
    site = SpinHalfSite(conserve=None)

    unit_cell_spin_lat = None
    basis = None

    if geometry == "YC":
        unit_cell_spin_lat = [[0.0, 0.0], [1.0, 0.0]]
        basis = [[2.0, 0.0], [0.5, sqrt(3) / 2.]]
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite", unit_cell=unit_cell_spin_lat,
                                            basis=basis, geometry=geometry)

    if state == "120":
        spin_state = Generate120DegOrderedState(triangular_lat, Lx, Ly, False)
    elif state == "stripe":
        spin_state = GenerateStripeOrderedState(triangular_lat, False)
    else:
        ValueError("Illegal spin state option")
        return

    print(spin_state.expectation_value("Sz"))

    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    PlotLattice(triangular_lat, ax_lat, plot_order=True)
    if local:
            plt.show()

    spin_corr_x = CalculateSpinSpinCorrelations(spin_state)
    ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lat)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_structure_factor(ks, spin_corr_k, triangular_lat, ax)
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite")
    triangular_lat.plot_brillouin_zone(ax)
    if local:
            plt.show()
    return spin_state


def TestZ2MeanFieldModel():
    from Main import Z2MeanFieldModel, AddCouplingsToZ2ModelDict, TestDictsAreCompatible

    triangular_lat = BuildTriangularLattice(3, 5, FermionSite(conserve="N"), "finite",
                                            bc=("open", "open"), spinfull_fermions=True)
    zeta = 5.0
    x_nn_hopping = 1.0
    y_nn_hopping = 2.0
    nnn_hopping = 3.0
    model_params = {"mu": 1.0, "zeta": 5.0, "init_H_MPO": False, "lattice": triangular_lat}
    model_params["hoppings"] = {(1, 0) : x_nn_hopping, (0, 1) : y_nn_hopping, (-1, 1) : x_nn_hopping,
                                (-1, 2): nnn_hopping, (1, 1): nnn_hopping}
    model_params["pairings"] = model_params["hoppings"]
    z2_model = Z2MeanFieldModel(model_params)

    center_sites = [14, 15]
    couplings_dict = PrintCouplings(z2_model, include_sites=center_sites)
    spindown_sites_coupled_to_spindown_center = [(13, (-1)*y_nn_hopping), (17, y_nn_hopping),
                                                 (5, -1.0*x_nn_hopping), (25, -1.0*x_nn_hopping), (7, -1.0*x_nn_hopping),
                                                 (23, -1.0*x_nn_hopping), (9, nnn_hopping), (21, nnn_hopping),
                                                 (3, (-1) * nnn_hopping), (27, (-1)*nnn_hopping)]

    expected_couplings_dict = AddCouplingsToZ2ModelDict(center_sites, spindown_sites_coupled_to_spindown_center, zeta)
    TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    tests_sites = [22, 23]
    couplings_dict = PrintCouplings(z2_model, include_sites=tests_sites)
    couplings_to_tests_sites = [(13, (-1) * x_nn_hopping), (15, (-1) * x_nn_hopping),
                                (25, (-1)*y_nn_hopping), (21, y_nn_hopping), (11, (-1)*nnn_hopping),
                                (17, (-1)*nnn_hopping)]
    expected_couplings_dict = AddCouplingsToZ2ModelDict(tests_sites, couplings_to_tests_sites, zeta)
    TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    print(z2_model.all_onsite_terms().to_TermList())
    CreateHamiltonianMatrixFromCouplingsList(z2_model, 30)
    fig, ax = plt.subplots()
    # PlotModelHoppingsByPhase(z2_model, ax)
    PlotLattice(triangular_lat, ax)
    plt.show()


def TestFreeFermionsSpinCorrelations():
    from Main import FreeFermionSpinCorrelations

    Lx, Ly = 6, 6
    spinfull_fermions = True

    site = FermionSite('N')
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite",
                                            ("open", "periodic"), "XC", spinfull_fermions)
    pi_flux_parameters = {"init_H_MPO": False, "monopole_Q": 0, "flux": 0.0,
                          "particle_hole": spinfull_fermions, "lattice": triangular_lat}
    C_x_exact, _ = CalculateExactCMatrixForPiFlux(0, pi_flux_parameters, model_type_dirac,
                                                  abs_magz=Lx*Ly)

    spin_spin_corr = FreeFermionSpinCorrelations(C_x_exact)
    spin_triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite", ("open", "periodic"),
                                                 "XC")
    momentum_space = True
    fig, ax = plt.subplots(figsize=(6, 5))
    if momentum_space:
        ks, C_k = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat)
        plot_structure_factor(ks, C_k, spin_triangular_lat, ax)
        lat_for_bz = BuildTriangularLattice(1, 1, site, "finite", ("open", "open"), "YC")
        lat_for_bz.plot_brillouin_zone(ax)
    else:
        ImshowMatrix(ax, fig, np.array([0.,1.]), np.array([0.,1.0]), spin_spin_corr, "X", "Y")

    plt.show()

    for i in range(spin_spin_corr.shape[0]):
        for j in range(spin_spin_corr.shape[1]):
            if i != j:
                assert(np.abs(spin_spin_corr[i,j] - 0.25) < 1e-14), "test failed - incorrect off diagonal term"
            else:
                assert(np.abs(spin_spin_corr[i,j] - 0.75) < 1e-14), "test failed - incorrect diagonal term"

def Test120State(Lx, Ly, geometry="YC"):
    from Main import getSpecielBzPoints

    fig, ax = plt.subplots()
    YC_lat = BuildTriangularLattice(1, 1, SpinHalfSite(None), "finite",
                              ("open", "open"), "YC")

    lat = BuildTriangularLattice(Lx, Ly, SpinHalfSite(None), "finite",
                              ("open", "periodic"), geometry)
    N = lat.N_sites
    psi = Generate120DegOrderedState(lat=lat, plot=False)
    spin_corr_x = CalculateSpinSpinCorrelations(psi)
    # spin_corr_x = np.loadtxt(code_dir + "../Meetings/spin_corr_x.csv")

    # b1, b2 = lat.reciprocal_basis
    ks, Sk = compute_structure_factor_grid(spin_corr_x, lat, wrap_displacements=True)

    # exit(1)

    if geometry == "XC":
        plot_structure_factor(ks, Sk, lat, ax=ax, mode='interpolate', n_tiles=1)
    else:
        plot_structure_factor(ks, Sk, lat, ax=ax, mode='voronoi', n_tiles=1)

    YC_lat.plot_brillouin_zone(ax, draw_points=False)
    plt.show()

    special_bz_points = getSpecielBzPoints()
    K, K_prime = special_bz_points["K"], special_bz_points["K_prime"]
    structure_factor_at_K = structure_factor(spin_corr_x, lat, K)
    structure_factor_at_K_prime = structure_factor(spin_corr_x, lat, K_prime)

    expected_max = ((1. / 4.) * N * (N / 3. - 1.) + (1. / 16.) * N * (2. * N / 3.) + 0.75 * N) / N
    expected_min = 0.5

    assert (np.abs(structure_factor_at_K_prime - expected_max) < 1e-14), "unexpected value for S in K point"
    assert (np.abs(structure_factor_at_K - expected_max) < 1e-14), "unexpected value for S in K point"

    if geometry != "XC":
        ks_minus_K_norm = np.linalg.norm(ks - K, axis=1)
        ks_minus_K_prime_norm = np.linalg.norm(ks - K_prime, axis=1)
        assert (np.min(ks_minus_K_norm) < 1e-14), "should have K point in k grid"
        assert (np.min(ks_minus_K_prime_norm) < 1e-14), "should have K' point in k grid"
        assert (np.abs(Sk[np.argmin(ks_minus_K_norm)] - expected_max) < 1e-14), "unexpected value for S in K point"
        assert (np.abs(Sk[np.argmin(ks_minus_K_prime_norm)] - expected_max) < 1e-14), "unexpected value for S in K' point"

    assert (np.abs(Sk[np.argmin(np.linalg.norm(ks, axis=1))] - expected_min) < 1e-14), \
        "unexpected value for S in Gamma point"

    print("Test finished succesfully")


def TestRandomStateCorrelations(geometry="YC", bc_MPS="finite"):
    from tenpy.algorithms.tebd import RandomUnitaryEvolution
    fig, ax = plt.subplots()
    Lx, Ly = 6, 6
    L = Lx * Ly
    spin_half = SpinHalfSite(conserve=None)
    psi = MPS.from_product_state([spin_half] * L, ["up", "down"] * (L // 2), bc=bc_MPS)
    options = dict(N_steps=1, trunc_params={'chi_max': 8}, dt=0.1)
    eng = RandomUnitaryEvolution(psi, options)
    eng.run()
    psi.canonical_form()
    lat = BuildTriangularLattice(Lx, Ly, SpinHalfSite(None), bc_MPS,
                                 ("periodic", "periodic"), geometry)
    spin_corr_x = CalculateSpinSpinCorrelations(psi)
    ks, Sk = compute_structure_factor_grid(spin_corr_x, lat, wrap_displacements=True)

    for kx in np.linspace(-pi, pi, 25):
        for ky in np.linspace(-pi, pi, 25):
            Kx, Ky = np.array([kx]), np.array([ky])
            _kx, _ky, sf_my_method = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat, Kx=Kx, Ky=Ky,
                                                                         new_implementation=False)
            sf_claude = structure_factor(spin_corr_x, lat, np.array([kx, ky]))
            diff_between_methods = np.abs(sf_claude - sf_my_method)
            if(diff_between_methods > 1e-13):
                print(f"Found large diff between methods: {diff_between_methods}")
                exit(1)

    plot_structure_factor(ks, Sk, lat, ax=ax, mode='interpolate', n_tiles=1)
    YC_lat = BuildTriangularLattice(1, 1, SpinHalfSite(None), "finite",
                                    ("open", "open"), "YC")
    YC_lat.plot_brillouin_zone(ax, draw_points=False)
    plt.show()


def StaticCorrelationsTests():
    Test120State(6, 6)
    Test120State(9, 9)
    Test120State(6, 3, "XC")
    Test120State(9, 4, "XC")
    TestRandomStateCorrelations(bc_MPS="infinite")
    TestRandomStateCorrelations()

