from WaveFunctionProperties import (plot_scalar_spin_chirality, compute_structure_factor_grid,
                                    plot_structure_factor, structure_factor, ComputeMomentumSpaceStructureFactor,
                                    CalculateSpinSpinCorrelations, CalculateDimerDimerCorrelations,
                                    _nearest_neighbor_dimer_bonds_by_site)
from numpy import sin, cos, sqrt, pi
import numpy as np
from tenpy.networks.site import FermionSite, SpinHalfSite
from Main import (BuildTriangularLattice, Generate120DegOrderedState, GenerateStripeOrderedState,
                  PrintCouplings, PlotLattice, model_type_dirac, ImshowMatrix)
from Gutzwiller import CreateHamiltonianMatrixFromCouplingsList, CalculateExactCMatrixForPiFlux
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
    lat = BuildTriangularLattice(3, 3, site, "finite", bc=("open", "open"), geometry="YC")
    psi = MPS.from_product_state(
        lat.mps_sites(),
        ["up"] * lat.N_sites,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )

    # all up: S_i . S_k = 1/4 on every bond, so <D_i D_j> = n_i n_j / 16 with n_i the number of bonds of site i
    sites = np.arange(lat.N_sites)
    n_bonds = np.array([len(bonds) for bonds in
                        _nearest_neighbor_dimer_bonds_by_site(psi, lat, sites).values()])
    dimer_corr = CalculateDimerDimerCorrelations(psi, lat)
    expected_dimer_corr = np.outer(n_bonds, n_bonds) / 16
    assert(dimer_corr.shape == expected_dimer_corr.shape)
    assert(np.max(np.abs(dimer_corr - expected_dimer_corr)) < 1e-13)

    partial_corr = CalculateDimerDimerCorrelations(psi, lat, sites1=[0, 4], sites2=[0, 4])
    assert(partial_corr.shape == (2, 2))
    assert(np.max(np.abs(partial_corr - expected_dimer_corr[np.ix_([0, 4], [0, 4])])) < 1e-13)
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
    elif state.startswith("stripe"):  # stripe0, stripe60 or stripe120
        spin_state = GenerateStripeOrderedState(triangular_lat, int(state[len("stripe"):]))
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
    from Gutzwiller import Z2MeanFieldModel, AddCouplingsToZ2ModelDict, TestDictsAreCompatible

    triangular_lat = BuildTriangularLattice(3, 5, FermionSite(conserve="N"), "finite",
                                            bc=("open", "open"), spinfull_fermions=True)
    zeta = 5.0
    x_nn_hopping = 1.0
    y_nn_hopping = 2.0
    nnn_hopping = 3.0
    bond_sign = -1
    model_params = {"mu": 1.0, "zeta": 5.0, "init_H_MPO": False, "lattice": triangular_lat, "bond_sign": bond_sign,
                    "flux": 0}
    model_params["hoppings"] = {(1, 0) : x_nn_hopping, (0, 1) : y_nn_hopping, (-1, 1) : x_nn_hopping,
                                (-1, 2): nnn_hopping, (1, 1): nnn_hopping, (-2, 1): nnn_hopping}
    model_params["pairings"] = model_params["hoppings"]
    z2_model = Z2MeanFieldModel(model_params)

    center_sites = [14, 15]
    couplings_dict = PrintCouplings(z2_model, include_sites=center_sites)
    # uniform (zero flux) ansatz: every bond has amplitude bond_sign * strength
    spindown_sites_coupled_to_spindown_center = [(13, bond_sign * y_nn_hopping), (17, bond_sign * y_nn_hopping),
                                                 (5, bond_sign * x_nn_hopping), (25, bond_sign * x_nn_hopping),
                                                 (7, bond_sign * x_nn_hopping), (23, bond_sign * x_nn_hopping),
                                                 (9, bond_sign * nnn_hopping), (21, bond_sign * nnn_hopping),
                                                 (3, bond_sign * nnn_hopping), (27, bond_sign * nnn_hopping)]

    expected_couplings_dict = AddCouplingsToZ2ModelDict(center_sites, spindown_sites_coupled_to_spindown_center, zeta)
    TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    tests_sites = [22, 23]
    couplings_dict = PrintCouplings(z2_model, include_sites=tests_sites)
    couplings_to_tests_sites = [(13, bond_sign * x_nn_hopping), (15, bond_sign * x_nn_hopping),
                                (25, bond_sign * y_nn_hopping), (21, bond_sign * y_nn_hopping),
                                (11, bond_sign * nnn_hopping), (17, bond_sign * nnn_hopping),
                                (5, bond_sign * nnn_hopping)]
    expected_couplings_dict = AddCouplingsToZ2ModelDict(tests_sites, couplings_to_tests_sites, zeta)
    TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    # pi flux through the cylinder: on a 3x4 cylinder (periodic y), the bonds of site (x, y) = (1, 3) (MPS sites 14, 15)
    # that cross the y boundary change sign for flux = 1, written as (x, y) -> (x', y') with y' taken mod Ly = 4:
    #   NN  dr = (0, 1):  (1, 3) -> (1, 0)   (MPS sites 8, 9)
    #   NN  dr = (-1, 1): (1, 3) -> (0, 0)   (MPS sites 0, 1)
    #   NNN dr = (1, 1):  (1, 3) -> (2, 0)   (MPS sites 16, 17)
    #   NNN dr = (-1, 2): (1, 3) -> (0, 1)   (MPS sites 2, 3)
    cylinder_lat = BuildTriangularLattice(3, 4, FermionSite(conserve="N"), "finite",
                                          bc=("open", "periodic"), spinfull_fermions=True)
    seam_sites = [14, 15]
    for flux in (0, 1):
        cylinder_params = dict(model_params, lattice=cylinder_lat, flux=flux)
        cylinder_model = Z2MeanFieldModel(cylinder_params)
        seam_sign = -1 if flux == 1 else 1
        couplings_dict = PrintCouplings(cylinder_model, include_sites=seam_sites)
        couplings_to_seam_sites = [(9, seam_sign * bond_sign * y_nn_hopping), (13, bond_sign * y_nn_hopping),
                                   (1, seam_sign * bond_sign * x_nn_hopping), (21, bond_sign * x_nn_hopping),
                                   (23, bond_sign * x_nn_hopping), (7, bond_sign * x_nn_hopping),
                                   (17, seam_sign * bond_sign * nnn_hopping), (5, bond_sign * nnn_hopping),
                                   (3, seam_sign * bond_sign * nnn_hopping), (19, bond_sign * nnn_hopping)]
        expected_couplings_dict = AddCouplingsToZ2ModelDict(seam_sites, couplings_to_seam_sites, zeta)
        TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    CreateHamiltonianMatrixFromCouplingsList(z2_model, 30)
    fig, ax = plt.subplots()
    PlotLattice(cylinder_lat, ax)
    plt.show()


def TestFreeFermionsSpinCorrelations():
    from Gutzwiller import FreeFermionSpinCorrelations

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
    from Main import getTriangularLatticeSpecielBzPoints

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

    special_bz_points = getTriangularLatticeSpecielBzPoints()
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


def HelicalUpSpinonHoppings(model_type, Lx, Ly, flux, helical=True):
    """ hopping matrix of the up spinons of the triangular mean field ansatz on the finite (helical) YC cylinder """
    from Gutzwiller import TriangularGutzwillerSpec
    spec = TriangularGutzwillerSpec("YC", model_type, helical=helical)
    lat = spec.fermion_lattice(FermionSite(conserve="N"), Lx, Ly, True, "finite")
    model_params = spec.mean_field_params(flux, 0, True)
    model_params["lattice"] = lat
    # asserts hermiticity
    H = CreateHamiltonianMatrixFromCouplingsList(spec.mean_field_model(model_params), lat.N_sites,
                                                 dtype=np.complex128)
    return H[0::2, 0::2] # the chain index n = x Ly + y of the spin lattice


def TestHelicalCylinder():
    """
        The helical YC cylinder YCLy-1 (Gutzwiller.helical_bc_shift): (x, y + Ly) ~ (x + 1, y), a chain n = x Ly + y with
        nearest neighbor bonds n -> n + 1, n + Ly - 1, n + Ly only.
    """
    from Gutzwiller import TriangularGutzwillerSpec, helical_bc_shift
    from Main import model_type_Z2, GutzwillerCaseDirName
    from WaveFunctionProperties import minimum_image_displacements, triangular_plaquettes
    tol = 1e-12

    for Lx, Ly in [(4, 4), (4, 5), (6, 3)]:
        N = Lx * Ly
        chain_bonds = {d: [(n, n + d) for n in range(N - d)] for d in (1, Ly - 1, Ly)}

        # the spin lattice: the bonds of the chain, in both the finite and the infinite MPS
        spin_lat = BuildTriangularLattice(Lx, Ly, SpinHalfSite(None), "finite", ("open", helical_bc_shift))
        bonds = set()
        for u1, u2, dr in spin_lat.pairs["nearest_neighbors"]:
            mps_i, mps_j, _, _ = spin_lat.possible_couplings(u1, u2, dr)
            bonds |= {(min(i, j), max(i, j)) for i, j in zip(mps_i, mps_j)}
        assert bonds == set(sum(chain_bonds.values(), [])), "helical lattice bonds are not those of the chain"
        inf_lat = BuildTriangularLattice(Lx, Ly, SpinHalfSite(None), "infinite", ("periodic", helical_bc_shift))
        for u1, u2, dr in inf_lat.pairs["nearest_neighbors"]:
            mps_i, mps_j, _, _ = inf_lat.possible_couplings(u1, u2, dr)
            assert len(mps_i) == N and set(np.abs(np.asarray(mps_j) - np.asarray(mps_i))) <= {1, Ly - 1, Ly}

        # the Dirac pi flux ansatz: hermitian, flux 0 through up and pi through down triangles, also on the seam
        t = HelicalUpSpinonHoppings(model_type_dirac, Lx, Ly, 0.0)
        for d, ds in chain_bonds.items():
            assert all(abs(abs(t[i, j]) - 1) < tol for i, j in ds), f"missing pi flux hopping along d = {d}"
        assert np.sum(np.abs(t) > tol) == 2 * sum(len(ds) for ds in chain_bonds.values())
        loop = lambda path: np.prod([t[path[k], path[(k + 1) % len(path)]] for k in range(len(path))])
        for n in range(N - Ly - 1):
            assert abs(loop([n, n + Ly, n + 1]) - 1) < tol, f"up triangle at {n} should have zero flux"
            assert abs(loop([n + 1, n + Ly, n + Ly + 1]) + 1) < tol, f"down triangle at {n} should have pi flux"

        # threaded flux: once on each loop around the helix n -> n + 1 -> ... -> n + Ly -> n, as on the YC cylinder
        flux = 0.3
        t_flux = HelicalUpSpinonHoppings(model_type_dirac, Lx, Ly, flux)
        around = lambda t, path: np.prod([t[path[k], path[k + 1]] for k in range(len(path) - 1)]) * t[path[-1], path[0]]
        for n in range(N - Ly):
            path = list(range(n, n + Ly + 1))
            assert abs(around(t_flux, path) / around(t, path) - np.exp(-1j * pi * flux)) < tol
        t_YC, t_YC_flux = (HelicalUpSpinonHoppings(model_type_dirac, Lx, Ly, f, helical=False) for f in (0.0, flux))
        path_YC = list(range(Ly, 2 * Ly))
        assert abs(around(t_YC_flux, path_YC) / around(t_YC, path_YC) - np.exp(-1j * pi * flux)) < tol

        # the Z2 ansatz is uniform, so its hoppings are translation invariant along the chain, except for the seam
        # sign of the vison (flux = 1)
        t_Z2 = HelicalUpSpinonHoppings(model_type_Z2, Lx, Ly, 0)
        for d, ds in chain_bonds.items():
            assert np.ptp([t_Z2[i, j] for i, j in ds]) < tol, f"Z2 hoppings along d = {d} are not uniform"

        # seam aware minimum image displacements and plaquettes
        rdisp = minimum_image_displacements(spin_lat, N)
        for d, ds in chain_bonds.items():
            for i, j in ds:
                assert abs(np.linalg.norm(rdisp[i, j]) - 1) < tol, f"bond {i}, {j} should have unit length"
        plaquettes = triangular_plaquettes(spin_lat)
        assert len(plaquettes) == 2 * (N - Ly) - 1
        for plaquette in plaquettes:
            i, j, k = plaquette["sites"]
            assert all(abs(np.linalg.norm(rdisp[a, b]) - 1) < tol for a, b in [(i, j), (j, k), (k, i)])

    # results directory, as generated from an input file (string parameters)
    dir_name = GutzwillerCaseDirName("r/", 2, 6, 1000, 0.0, "YC", "infinite", 0, model_type_dirac, 0.0, 0,
                                     helical="True")
    assert "_YC6-1_" in dir_name, dir_name
    assert "_YC_" in GutzwillerCaseDirName("r/", 2, 6, 1000, 0.0, "YC", "infinite", 0, model_type_dirac, 0.0, 0,
                                           helical="False")
    print("TestHelicalCylinder finished successfully")


def HelicalProjectedBondCorrelations(model_type, Ly, flux, virtual_parities, Lx=2, chi_max=600):
    """ <Sz_n Sz_n+d> for d = 1, Ly of the projected iMPS on the helical cylinder, for each virtual parity """
    from Gutzwiller import (TriangularGutzwillerSpec, GetFluxSlaterMPS, RescaleMPSForGutzwiller, iMPSAbrikosovPH,
                            svd_min_slater_default)
    trunc_par = {"chi_max": chi_max, "svd_min": svd_min_slater_default, "degeneracy_tol": 1e-12}
    psi_slater, _, _ = GetFluxSlaterMPS(TriangularGutzwillerSpec("YC", model_type, helical=True), Lx, Ly, True,
                                        FermionSite(conserve="N"), trunc_par, "infinite", 0, flux=flux)
    psi_slater.canonical_form()
    RescaleMPSForGutzwiller(psi_slater)
    correlations = {}
    for virtual_parity in virtual_parities:
        psi = psi_slater.copy()
        iMPSAbrikosovPH(psi, virtual_parity)
        psi.canonical_form()
        correlations[virtual_parity] = np.array([[np.real(psi.correlation_function("Sz", "Sz", sites1=[n],
                                                                                   sites2=[n + d])[0, 0])
                                                  for n in range(psi.L)] for d in (1, Ly)])
    return correlations


def TestHelicalProjectionTranslation(Ly=3, tol=1e-6):
    """
        On the helical cylinder, the shift by one site T maps the virtual parity sector p of the projected iMPS to 1 - p
        (each projected site changes 2 Sz of the left half by +-1), and for the Dirac pi flux ansatz also the spinon
        flux theta to theta + pi (neighboring loops around the helix differ by a pi flux rhombus). So a fixed sector
        is only invariant under a shift by two sites, and T relates the sectors:
            Z2:    <O>_(theta, p)(n) = <O>_(theta, 1 - p)(n + 1)
            Dirac: <O>_(theta, p)(n) = <O>_(theta + pi, 1 - p)(n + 1)
    """
    from Main import model_type_Z2
    shift = lambda corr: np.roll(corr, -1, axis=1)

    z2 = HelicalProjectedBondCorrelations(model_type_Z2, Ly, 0, (0, 1))
    assert np.max(np.abs(z2[0] - shift(z2[1]))) < tol, "Z2: T should map virtual parity 0 to 1"
    assert np.max(np.abs(z2[1] - np.roll(z2[1], -2, axis=1))) < tol, "Z2: a sector should be invariant under T^2"
    assert np.max(np.abs(z2[0] - z2[1])) > 1e3 * tol, "Z2: expected locally distinct sectors at this Ly"

    dirac_0 = HelicalProjectedBondCorrelations(model_type_dirac, Ly, 0.0, (0, 1))
    dirac_pi = HelicalProjectedBondCorrelations(model_type_dirac, Ly, 1.0, (0, 1))
    for p in (0, 1):
        assert np.max(np.abs(dirac_0[p] - shift(dirac_pi[1 - p]))) < tol, \
            f"Dirac: T should map (flux 0, virtual parity {p}) to (flux pi, virtual parity {1 - p})"
    print("TestHelicalProjectionTranslation finished successfully")


def StaticCorrelationsTests():
    Test120State(6, 6)
    Test120State(9, 9)
    Test120State(6, 3, "XC")
    Test120State(9, 4, "XC")
    TestRandomStateCorrelations(bc_MPS="infinite")
    TestRandomStateCorrelations()


if __name__ == "__main__":
    TestHelicalCylinder()
    TestHelicalProjectionTranslation()
    TestZ2MeanFieldModel()