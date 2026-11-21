"""Checks of the per-site energies (WaveFunctionProperties.SiteEnergies, BulkSites; Main.calculateSiteEnergiesJ1J2)
and of the bulk energy of the Gutzwiller energy post-process (PostProcess._CalculateGutzwillerEnergy).

- The site energies add up to <H> (finite MPS) or average to the energy per site (infinite MPS), also with the J2
  bonds that coincide on Ly = 3 and on the Kagome lattice.
- They agree with half the bonds on each site, summed from the full correlation matrices over the lattice pairs.
- In the fully polarized state each bulk site has E_i = 3 Delz (J1 + J2) / 4 (6 + 6 bonds of <S.S> = Delz / 4,
  halved); the open edges have less.
- The post-process saves the bulk energy and the site energies, and the collection pads older cases with nan.
"""
import pickle

import numpy as np
import pytest
from tenpy.algorithms.tebd import RandomUnitaryEvolution
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

from Main import BuildKagomeLattice, BuildTriangularLattice, GenerateJ1J2SpinModel, calculateSiteEnergiesJ1J2
from WaveFunctionProperties import BulkEnergyPerSite, BulkSites, SiteEnergies

SITE = SpinHalfSite(conserve="Sz")
FINITE_BC = ("open", "periodic")
INFINITE_BC = ("periodic", "periodic")


def Triangular(Lx, Ly, bc_MPS, geometry="YC", site=SITE, **kwargs):
    bc = FINITE_BC if bc_MPS == "finite" else INFINITE_BC
    return BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc, geometry=geometry, **kwargs)


def Kagome(Lx, Ly, bc_MPS, site=SITE):
    return BuildKagomeLattice(Lx, Ly, site, bc_MPS, FINITE_BC if bc_MPS == "finite" else INFINITE_BC)


def RandomState(lat, seed, n_steps=12, chi_max=24):
    """A generic entangled Sz = 0 state: random unitary circuits on a Neel product state in the MPS order."""
    np.random.seed(seed)
    psi = MPS.from_product_state(lat.mps_sites(), ["up", "down"] * (lat.N_sites // 2) + ["up"] * (lat.N_sites % 2),
                                 bc=lat.bc_MPS)
    RandomUnitaryEvolution(psi, {"N_steps": n_steps, "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12}}).run()
    psi.canonical_form()
    return psi


def PolarizedState(lat):
    return MPS.from_product_state(lat.mps_sites(), ["up"] * lat.N_sites, bc=lat.bc_MPS)


def EnergyPerSiteFromSites(psi, site_energies):
    return np.sum(site_energies) / psi.L if psi.finite else np.mean(site_energies)


# --- the sum rule
@pytest.mark.parametrize("make_lattice", [
    lambda: Triangular(4, 3, "finite"),
    lambda: Triangular(4, 4, "finite"),
    lambda: Triangular(3, 4, "finite", geometry="XC"),
    lambda: Triangular(2, 3, "infinite"),
    lambda: Triangular(2, 4, "infinite"),
    lambda: Kagome(2, 2, "finite"),
    lambda: Kagome(1, 2, "infinite"),
], ids=["YC_finite_Ly3", "YC_finite_Ly4", "XC_finite", "YC_infinite_Ly3", "YC_infinite_Ly4", "Kagome_finite",
        "Kagome_infinite"])
@pytest.mark.parametrize("J2, Delz", [(0.0, 1.0), (0.3, 0.8)])
def test_site_energies_sum_to_mpo_energy(make_lattice, J2, Delz):
    lat = make_lattice()
    psi = RandomState(lat, seed=1)
    model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
    E_mpo = model.H_MPO.expectation_value(psi)
    E_mpo = E_mpo / psi.L if psi.finite else E_mpo
    site_energies = SiteEnergies(psi, model)
    assert len(site_energies) == psi.L
    np.testing.assert_allclose(EnergyPerSiteFromSites(psi, site_energies), E_mpo, atol=1e-10)
    # a nontrivial check: the state is not translation invariant, so the sites differ
    assert np.ptp(site_energies) > 1e-3


# --- an independent evaluation from the lattice pairs
def test_site_energies_match_half_bonds_from_correlation_matrices():
    J2, Delz = 0.4, 0.7
    lat = Triangular(5, 4, "finite")
    psi = RandomState(lat, seed=2)
    zz = psi.correlation_function("Sz", "Sz")
    pm = psi.correlation_function("Sp", "Sm")
    expected = np.zeros(psi.L)
    for pair_key, J in (("nearest_neighbors", 1.0), ("next_nearest_neighbors", J2)):
        for u1, u2, dx in lat.pairs[pair_key]:
            mps_i, mps_j, _, _ = lat.possible_couplings(u1, u2, dx)
            for i, j in zip(mps_i, mps_j):
                bond = J * (Delz * zz[i, j] + 0.5 * (pm[i, j] + pm[j, i]))
                expected[i] += 0.5 * np.real(bond)
                expected[j] += 0.5 * np.real(bond)
    model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
    np.testing.assert_allclose(SiteEnergies(psi, model), expected, atol=1e-10)


# --- the polarized state: the bulk value is exact, the open edges differ
@pytest.mark.parametrize("bc_MPS, Lx, Ly", [("finite", 6, 4), ("finite", 5, 3), ("infinite", 2, 3), ("infinite", 1, 4)])
def test_polarized_state_bulk_energy(bc_MPS, Lx, Ly):
    J2, Delz = 0.25, 0.6
    lat = Triangular(Lx, Ly, bc_MPS)
    model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
    site_energies = SiteEnergies(PolarizedState(lat), model)
    E_bulk = 3 * Delz * (1.0 + J2) / 4
    bulk = BulkSites(lat)
    np.testing.assert_allclose(site_energies[bulk], E_bulk, atol=1e-12)
    assert np.isclose(BulkEnergyPerSite(site_energies, lat), E_bulk)
    if bc_MPS == "finite":  # the edge columns miss bonds
        edge = [i for i in range(lat.N_sites) if lat.mps2lat_idx(i)[0] in (0, Lx - 1)]
        assert np.all(site_energies[edge] < E_bulk - 1e-3)


# --- the J2 split and the shared cache
def test_J1_J2_split_and_cache():
    Delz = 0.9
    lat = Triangular(4, 4, "finite")
    psi = RandomState(lat, seed=3)
    E_J1, E_J2 = calculateSiteEnergiesJ1J2(psi, lat, Delz)
    for J2 in (0.0, 0.125, 0.37, 1.0):
        model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
        np.testing.assert_allclose(E_J1 + J2 * E_J2, SiteEnergies(psi, model), atol=1e-10)
    # a cache filled by another model gives the same values
    cache = {}
    SiteEnergies(psi, GenerateJ1J2SpinModel(1.0, Delz, lat)[0], cache)
    model, _ = GenerateJ1J2SpinModel(0.37, Delz, lat)
    np.testing.assert_allclose(SiteEnergies(psi, model, cache), SiteEnergies(psi, model), atol=1e-12)


# --- the bulk sites
def test_bulk_sites():
    def columns(lat):
        return sorted({int(lat.mps2lat_idx(i)[0]) for i in BulkSites(lat)})

    assert columns(Triangular(6, 4, "finite")) == [2, 3]  # even Lx: the two central columns
    assert columns(Triangular(5, 4, "finite")) == [2]  # odd Lx: the central column
    assert len(BulkSites(Triangular(6, 4, "finite"))) == 2 * 4
    assert BulkSites(Triangular(2, 3, "infinite")).tolist() == list(range(6))
    lat = Triangular(6, 4, "finite", remove_sites="edges")  # the removed sites are in the edge columns
    assert len(BulkSites(lat)) == 2 * 4
    assert columns(Kagome(4, 2, "finite")) == [1, 2]


# --- the post-process
def _write_gutzwiller_case(results, name, psi, lat):
    from ResultsIO import MPS_LATTICE_FILE

    case_dir, pkl_dir = results / name, results / ("pklFiles_" + name)
    case_dir.mkdir(parents=True)
    pkl_dir.mkdir(parents=True)
    with open(pkl_dir / "psi_gutzwiller.pkl", "wb") as f:
        pickle.dump(psi, f)
    with open(pkl_dir / MPS_LATTICE_FILE, "wb") as f:
        pickle.dump(lat, f)
    return case_dir


@pytest.mark.parametrize("bc_MPS, Lx", [("finite", 6), ("infinite", 2)])
def test_gutzwiller_energy_post_process(tmp_path, bc_MPS, Lx):
    from PostProcess import (CalculateTriangularGutzwillerEnergy, CollectPostProcessResults, TRIANGULAR_ENERGIES_FILE,
                             TRIANGULAR_GUTZWILLER_ENERGY, TRIANGULAR_SITE_ENERGIES_FILE)

    Delz, J2s = 1.0, np.array([0.0, 0.1, 0.2])
    lat = Triangular(Lx, 4, bc_MPS)
    psi = RandomState(lat, seed=4)
    results = tmp_path / "results"
    case_dir = _write_gutzwiller_case(results, "case", psi, lat)
    np.savetxt(results / "J2.txt", J2s)

    CalculateTriangularGutzwillerEnergy(str(case_dir) + "/", J2_file="J2.txt", Delz=Delz)

    energies = np.loadtxt(case_dir / TRIANGULAR_ENERGIES_FILE, ndmin=2)
    assert energies.shape == (len(J2s), 3)
    np.testing.assert_allclose(energies[:, 0], J2s)
    site_file = np.loadtxt(case_dir / TRIANGULAR_SITE_ENERGIES_FILE, ndmin=2)
    assert site_file.shape == (lat.N_sites, 1 + 3 + 2)  # mps index, (x, y, u), E_J1, E_J2
    np.testing.assert_array_equal(site_file[:, 0], np.arange(lat.N_sites))
    E_J1, E_J2 = site_file[:, -2], site_file[:, -1]
    for J2, E, E_bulk in energies:
        model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
        E_mpo = model.H_MPO.expectation_value(psi)
        np.testing.assert_allclose(E, E_mpo / psi.L if psi.finite else E_mpo, atol=1e-10)
        np.testing.assert_allclose(E_bulk, np.mean((E_J1 + J2 * E_J2)[BulkSites(lat)]), atol=1e-12)
        np.testing.assert_allclose(E_J1 + J2 * E_J2, SiteEnergies(psi, model), atol=1e-10)
    if bc_MPS == "infinite":  # the whole unit cell is the bulk
        np.testing.assert_allclose(energies[:, 2], energies[:, 1], atol=1e-10)

    # a case post-processed before the bulk energy column is collected with nan there
    legacy_dir = results / "legacy"
    legacy_dir.mkdir()
    np.savetxt(legacy_dir / TRIANGULAR_ENERGIES_FILE, np.column_stack([J2s, -0.5 * np.ones_like(J2s)]))
    out = tmp_path / "collected.txt"
    CollectPostProcessResults(results, TRIANGULAR_GUTZWILLER_ENERGY, out)
    collected = np.loadtxt(out, ndmin=2)
    assert collected.shape == (2 * len(J2s), 4)
    with open(out) as f:
        assert f.readline().split()[1:] == ["dir_index", "J2", "energy_per_site", "bulk_energy_per_site"]
    dirs = [line.split()[1] for line in open(str(out) + ".dirs.txt")]
    legacy_index = [ind for ind, d in enumerate(dirs) if d.endswith("legacy")][0]
    legacy_rows = collected[collected[:, 0] == legacy_index]
    assert np.all(np.isnan(legacy_rows[:, 3]))
    np.testing.assert_allclose(collected[collected[:, 0] != legacy_index][:, 1:], energies)


# --- the DMRG ground states (PostProcess.CalculateSiteEnergies, post-process type site_energies)
def test_dmrg_case_parameters():
    from PostProcess import _dmrg_case_parameters

    assert _dmrg_case_parameters("a/finite_init_Random_conserve_True_J2_0.0_Delz_2.0_chi_400_maxsweeps_10/") == \
        (0.0, 2.0)
    assert _dmrg_case_parameters("infinite_init_120_conserve_True_J2_0.125_chi_800") == (0.125, 1.0)
    assert _dmrg_case_parameters("finite_init_Random_conserve_True_J2_0.06_Delz_0.7_magz_0.0100") == (0.06, 0.7)
    with pytest.raises(AssertionError):
        _dmrg_case_parameters("Lx_6_Ly_4_bc_op_YC")


def _write_dmrg_case(results, geometry_dir, name, psi, lat_pkl):
    """results/<geometry_dir>/<name>/ with run.log, and its pklFiles_<name>/ sibling with psi_gs.pkl and lattice.pkl
    (as Main.J1J2DMRG saves them)."""
    case_dir = results / geometry_dir / name
    pkl_dir = results / geometry_dir / ("pklFiles_" + name)
    case_dir.mkdir(parents=True)
    pkl_dir.mkdir(parents=True)
    (case_dir / "run.log").write_text("")
    with open(pkl_dir / "psi_gs.pkl", "wb") as f:
        pickle.dump(psi, f)
    with open(pkl_dir / "lattice.pkl", "wb") as f:
        pickle.dump(lat_pkl, f)
    return case_dir


NO_CONSERVE = SpinHalfSite(conserve=None)


# (MPS lattice, lattice.pkl): the same for a finite run, the correlations lattice (more columns) for an infinite one
@pytest.mark.parametrize("make_lattices", [
    lambda: (Triangular(6, 4, "finite"),) * 2,
    lambda: (Triangular(6, 4, "finite", remove_sites="edges"),) * 2,
    lambda: (Kagome(3, 2, "finite"),) * 2,
    lambda: (Triangular(2, 4, "infinite"), Triangular(8, 4, "infinite")),
    lambda: (Triangular(2, 3, "infinite", geometry="XC"), Triangular(6, 3, "infinite", geometry="XC")),
    lambda: (Triangular(2, 3, "infinite", site=NO_CONSERVE), Triangular(6, 3, "infinite", site=NO_CONSERVE)),
    lambda: (Kagome(1, 2, "infinite"), Kagome(4, 2, "infinite")),
], ids=["YC_finite", "YC_finite_removed_edges", "Kagome_finite", "YC_infinite", "XC_infinite",
        "YC_infinite_no_conserve", "Kagome_infinite"])
def test_dmrg_site_energies_post_process(tmp_path, make_lattices):
    from PostProcess import (PostProcessResults, SITE_ENERGIES, DMRG_BULK_ENERGY_FILE, DMRG_SITE_ENERGIES_FILE,
                             _dmrg_mps_lattice)

    J2, Delz = 0.15, 0.8
    lat, lat_pkl = make_lattices()
    psi = RandomState(lat, seed=5)
    case_dir = _write_dmrg_case(tmp_path / "results", "geometry",
                                f"{lat.bc_MPS}_init_Random_conserve_True_J2_{J2}_Delz_{Delz}_chi_64", psi, lat_pkl)

    # the MPS lattice is recovered from lattice.pkl
    lat_rebuilt = _dmrg_mps_lattice(psi, str(case_dir) + "/")
    assert lat_rebuilt.N_sites == psi.L
    np.testing.assert_allclose([lat_rebuilt.position(lat_rebuilt.mps2lat_idx(i)) for i in range(psi.L)],
                               [lat.position(lat.mps2lat_idx(i)) for i in range(psi.L)])

    PostProcessResults(str(case_dir), SITE_ENERGIES)

    model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
    expected = SiteEnergies(psi, model)
    E_mpo = model.H_MPO.expectation_value(psi)
    E_mpo = E_mpo / psi.L if psi.finite else E_mpo
    site_file = np.loadtxt(case_dir / DMRG_SITE_ENERGIES_FILE, ndmin=2)
    assert site_file.shape == (psi.L, 1 + 3 + 1)
    np.testing.assert_array_equal(site_file[:, 1:4], [lat.mps2lat_idx(i) for i in range(psi.L)])
    np.testing.assert_allclose(site_file[:, -1], expected, atol=1e-10)
    (J2_saved, Delz_saved, E, E_bulk), = np.loadtxt(case_dir / DMRG_BULK_ENERGY_FILE, ndmin=2)
    assert (J2_saved, Delz_saved) == (J2, Delz)
    np.testing.assert_allclose(E, E_mpo, atol=1e-10)
    np.testing.assert_allclose(E_bulk, np.mean(expected[BulkSites(lat)]), atol=1e-10)
    if psi.finite:
        assert not np.isclose(E_bulk, E)  # the bulk is a subset of the sites, not their mean
    else:
        np.testing.assert_allclose(E_bulk, E, atol=1e-10)


def test_dmrg_bulk_energy_collection(tmp_path):
    from PostProcess import CollectPostProcessResults, PostProcessResults, SITE_ENERGIES

    results = tmp_path / "results"
    cases = [("finite", 0.0, 1.0), ("finite", 0.1, 0.7)]
    expected = []
    for ind, (bc_MPS, J2, Delz) in enumerate(cases):
        lat = Triangular(4, 3, bc_MPS)
        psi = RandomState(lat, seed=10 + ind)
        name = f"{bc_MPS}_init_Random_conserve_True_J2_{J2}" + ("" if Delz == 1.0 else f"_Delz_{Delz}") + "_chi_32"
        PostProcessResults(str(_write_dmrg_case(results, "geometry", name, psi, lat)), SITE_ENERGIES)
        model, _ = GenerateJ1J2SpinModel(J2, Delz, lat)
        expected.append([J2, Delz, model.H_MPO.expectation_value(psi) / psi.L])
    out = tmp_path / "collected.txt"
    CollectPostProcessResults(results, SITE_ENERGIES, out)
    collected = np.loadtxt(out, ndmin=2)
    with open(out) as f:
        assert f.readline().split()[1:] == ["dir_index", "J2", "Delz", "energy_per_site", "bulk_energy_per_site"]
    assert collected.shape == (len(cases), 5)
    dirs = [line.split(maxsplit=1)[1].strip() for line in open(str(out) + ".dirs.txt")]
    for row in collected:
        J2, Delz, E = expected[[ind for ind, (_, J2, _) in enumerate(cases) if f"_J2_{J2}_" in dirs[int(row[0])]][0]]
        np.testing.assert_allclose(row[1:4], [J2, Delz, E], atol=1e-10)


@pytest.mark.parametrize("bc_MPS, Lx", [("finite", 8), ("infinite", 2)])
def test_dmrg_run_and_site_energies(tmp_path, monkeypatch, bc_MPS, Lx):
    """The site_energies post-process of a ground state saved by Main.J1J2DMRG itself (for an infinite run, its
    lattice.pkl is the longer correlations lattice)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import Main
    from PostProcess import PostProcessResults, SITE_ENERGIES, DMRG_BULK_ENERGY_FILE
    from ResultsIO import LoadLattice, LoadPsi

    J2, Delz = 0.1, 0.9
    Ly = 3 if bc_MPS == "finite" else 4  # Ly > Lx: a longer correlations lattice (Main.LxInfiniteMPSCorrelations)
    case_dir = tmp_path / f"{bc_MPS}_init_Random_conserve_True_J2_{J2}_Delz_{Delz}_chi_64"
    case_dir.mkdir()
    monkeypatch.setattr(Main, "local", False)  # results in the working directory, no plt.show()
    monkeypatch.chdir(case_dir)
    bc = "open-periodic" if bc_MPS == "finite" else "periodic-periodic"
    Main.TriangularJ1J2DMRG(Lx, Ly, bc, bc_MPS, geometry="YC", conserve=True, initial_state="Random", J2=J2,
                            chi_max=64, max_sweeps=8, Delz=Delz, ramp=False)
    plt.close("all")
    psi = LoadPsi(case_dir, "psi_gs.pkl")
    if bc_MPS == "infinite":
        assert LoadLattice(case_dir).N_sites > psi.L

    PostProcessResults(str(case_dir), SITE_ENERGIES)

    (_, _, E, E_bulk), = np.loadtxt(case_dir / DMRG_BULK_ENERGY_FILE, ndmin=2)
    E_dmrg = np.loadtxt(case_dir / "Energies.txt")[-1]  # total for a finite run, per site for an infinite one
    np.testing.assert_allclose(E, E_dmrg / psi.L if psi.finite else E_dmrg, atol=1e-6)
    if psi.finite:  # the edge sites miss bonds: the bulk is lower than the mean
        assert E_bulk < E - 1e-3
    else:
        np.testing.assert_allclose(E_bulk, E, atol=1e-10)
