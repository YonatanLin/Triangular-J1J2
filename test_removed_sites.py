"""Checks of the triangular J1J2 DMRG on a finite YC cylinder with removed sites (Main.ParseRemovedSites, the odd
sector of Hu et al., PRB 92, 140403(R) (2015), Fig. 2(b)).

- "edges" removes (0, 0) and (Lx - 1, Ly - 1): one site of each edge column, exchanged by the inversion through the
  center of the cylinder; other values are refused.
- The couplings of the model are those of the full cylinder without the ones touching a removed site (compared by
  lattice index, since a removed site has no MPS index).
- The DMRG ground state energy agrees with exact diagonalization.
- The whole DMRG run (J1J2DMRG) and its post-processing (entanglement entropy, Jiang analysis) run on it.
- Gutzwiller: a removed spin site is both fermion flavors of it, whose (up, down) pairs follow the MPS order of the spin
  lattice; the mean field couplings are those of the full cylinder without the removed sites (also the monopole
  Landau gauge, whose flux per plaquette doesn't change); the projected state and its energy post-process.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

import Main
from Main import BuildTriangularLattice, GenerateJ1J2SpinModel, ParseRemovedSites, TriangularGeometryDirName
from ResultsIO import LoadLattice, LoadPsi
from WaveFunctionProperties import (HasRemovedSites, LatticeSiteExists, ColumnCutBonds, CentralColumnCutBond,
                                    triangular_plaquettes)

BC = ("open", "periodic")


def EdgesLattice(Lx, Ly, conserve="Sz"):
    return BuildTriangularLattice(Lx, Ly, SpinHalfSite(conserve), "finite", BC, "YC", remove_sites="edges")


def ColumnLengths(lat):
    columns = np.array([lat.mps2lat_idx(i)[0] for i in range(lat.N_sites)])
    return np.bincount(columns, minlength=lat.Ls[0])


# --- the option
def test_parse_removed_sites():
    assert ParseRemovedSites(None, 6, 4, "YC") is None
    assert ParseRemovedSites("None", 6, 4, "YC") is None
    assert ParseRemovedSites("edges", 6, 4, "YC") == [[0, 0, 0], [5, 3, 0]]
    with pytest.raises(ValueError):
        ParseRemovedSites("0.0-5.3", 6, 4, "YC")
    with pytest.raises(ValueError):
        ParseRemovedSites("edges", 6, 4, "XC")
    with pytest.raises(ValueError):  # removing sites needs a finite MPS
        BuildTriangularLattice(6, 4, SpinHalfSite("Sz"), "infinite", ("periodic", "periodic"), "YC", remove_sites="edges")
    assert not HasRemovedSites(BuildTriangularLattice(6, 4, SpinHalfSite("Sz"), "finite", BC, "YC", remove_sites="None"))


def test_geometry_dir_name():
    assert TriangularGeometryDirName(24, 6, BC, "YC") == "Lx_24_Ly_6_bc_op_YC/"
    assert TriangularGeometryDirName(24, 6, BC, "YC", "None") == "Lx_24_Ly_6_bc_op_YC/"
    assert TriangularGeometryDirName("24", "6", BC, "YC", "edges") == "Lx_24_Ly_6_bc_op_YC_rm_edges/"


@pytest.mark.parametrize("Lx, Ly", [(6, 4), (8, 6), (5, 8)])
def test_odd_sector_lattice(Lx, Ly):
    lat = EdgesLattice(Lx, Ly)
    assert HasRemovedSites(lat) and lat.N_sites == Lx * Ly - 2
    # one site less in each edge column: an odd number of sites in the first column for even Ly
    assert ColumnLengths(lat).tolist() == [Ly - 1] + [Ly] * (Lx - 2) + [Ly - 1]
    removed = {tuple(r) for r in lat.remove.tolist()}
    assert removed == {(0, 0, 0), (Lx - 1, Ly - 1, 0)}
    assert all(not LatticeSiteExists(lat, r) for r in removed)
    # the removed sites are exchanged by the inversion through the center
    assert {(Lx - 1 - x, Ly - 1 - y, u) for x, y, u in removed} == removed
    # the cuts between columns and the central one (tenpy bond convention: left of its site)
    cuts = ColumnCutBonds(lat)
    assert (np.cumsum(ColumnLengths(lat))[:-1] - 1).tolist() == cuts.tolist()
    assert CentralColumnCutBond(lat) == (Lx // 2) * Ly - 1
    # no chirality plaquette on a removed site
    for plaquette in triangular_plaquettes(lat):
        assert all(0 <= i < lat.N_sites for i in plaquette["sites"])


# --- the couplings (Q: how to check a removed site isn't coupled, when it has no MPS index)
def CouplingsByLatticeSite(model):
    """{((op, lattice index), ...): strength} of the coupling terms of model, keyed by lattice index, not MPS index"""
    couplings = {}
    for term, strength in model.all_coupling_terms().to_TermList():
        key = tuple((op, tuple(int(v) for v in model.lat.mps2lat_idx(i))) for op, i in term)
        couplings[key] = couplings.get(key, 0.0) + strength
    return couplings


@pytest.mark.parametrize("Lx, Ly, J2", [(6, 4, 0.1), (5, 6, 0.125)])
def test_couplings_are_full_cylinder_minus_removed_sites(Lx, Ly, J2):
    lat = EdgesLattice(Lx, Ly)
    full_lat = BuildTriangularLattice(Lx, Ly, SpinHalfSite("Sz"), "finite", BC, "YC")
    removed = {tuple(r) for r in lat.remove.tolist()}

    couplings = CouplingsByLatticeSite(GenerateJ1J2SpinModel(J2, 1.0, lat)[0])
    full_couplings = CouplingsByLatticeSite(GenerateJ1J2SpinModel(J2, 1.0, full_lat)[0])
    expected = {key: s for key, s in full_couplings.items() if not any(site in removed for _, site in key)}
    assert len(expected) < len(full_couplings)  # the removed sites had couplings
    assert couplings.keys() == expected.keys()
    assert all(np.isclose(couplings[key], expected[key]) for key in expected)


# --- DMRG vs exact diagonalization
def test_dmrg_energy_matches_exact_diagonalization():
    J2 = 0.1
    lat = EdgesLattice(4, 4)  # 14 sites
    model, _ = GenerateJ1J2SpinModel(J2, 1.0, lat)

    # max_size is checked against the full 2^14 x 2^14, but H is block sparse in Sz (largest block 3432 x 3432)
    ed = ExactDiag(model, charge_sector=[0], max_size=3e8)
    ed.build_full_H_from_mpo()
    ed.full_diagonalization()
    E_ed = np.min(ed.E)

    psi = MPS.from_product_state(lat.mps_sites(), ["up", "down"] * (lat.N_sites // 2), bc="finite")
    info = dmrg.run(psi, model, {"mixer": True, "trunc_params": {"chi_max": 128, "svd_min": 1e-12},
                                 "max_E_err": 1e-12, "min_sweeps": 4, "max_sweeps": 20})
    assert abs(info["E"] - E_ed) < 1e-8, (info["E"], E_ed)


# --- the DMRG run and its post-processing
def test_dmrg_run_and_post_process(tmp_path, monkeypatch):
    from PostProcess import CalculateCentralBondEntanglementEntropy, EE_FILE
    from JiangAnalysis import AnalyzeCaseAsJiang, J1J2Energies, PlotBondCorrelationsVsLx

    Lx, Ly, J2 = 8, 4, 0.1
    monkeypatch.setattr(Main, "local", False)  # results in the working directory, no plt.show()
    monkeypatch.chdir(tmp_path)
    Main.TriangularJ1J2DMRG(Lx, Ly, "open-periodic", "finite", geometry="YC", remove_sites="edges", conserve=True,
                            initial_state="Random", J2=J2, chi_max=40, max_sweeps=8)

    lat, psi = LoadLattice(tmp_path), LoadPsi(tmp_path, "psi_gs.pkl")
    assert HasRemovedSites(lat) and lat.N_sites == psi.L == Lx * Ly - 2
    E_per_site = np.loadtxt(tmp_path / "Energies.txt")[-1] / psi.L

    CalculateCentralBondEntanglementEntropy(str(tmp_path) + "/")
    central_bond = int(np.loadtxt(tmp_path / EE_FILE)[0])
    assert central_bond == CentralColumnCutBond(lat) == (Lx // 2) * Ly - 1

    _, lxs, _ = PlotBondCorrelationsVsLx(tmp_path, save=False)
    assert 0 < len(lxs) and min(lxs) >= 2 and max(lxs) <= Lx - 3  # the columns of and next to the removed sites

    assert np.isclose(J1J2Energies(psi, tmp_path, J2s=[J2])[0], E_per_site)
    AnalyzeCaseAsJiang(psi, tmp_path, J2s=[J2])
    plt.close("all")


# --- the Gutzwiller projection (a removed spin site is both fermion flavors of it)
def GutzwillerSpec(model_type="Dirac", helical=False, remove_sites="edges"):
    from Gutzwiller import TriangularGutzwillerSpec
    return TriangularGutzwillerSpec("YC", model_type, helical=helical, remove_sites=remove_sites)


@pytest.mark.parametrize("helical", [False, True])
def test_fermion_lattice_pairs_follow_spin_lattice(helical):
    from tenpy.networks.site import FermionSite

    Lx, Ly = 6, 4
    spec = GutzwillerSpec(helical=helical)
    fermion_lat = spec.fermion_lattice(FermionSite("N"), Lx, Ly, True, "finite")
    spin_lat = spec.spin_lattice(SpinHalfSite("Sz"), Lx, Ly, "finite", fermion_lat.boundary_conditions)
    assert HasRemovedSites(fermion_lat) and HasRemovedSites(spin_lat)
    assert fermion_lat.N_sites == 2 * spin_lat.N_sites == 2 * (Lx * Ly - 2)
    for x, y, u in spin_lat.remove.tolist():
        assert not any(LatticeSiteExists(fermion_lat, [x, y, 2 * u + spin]) for spin in (0, 1))
    # the projection pairs the fermion MPS sites 2i (up), 2i + 1 (down) into the spin MPS site i
    for i in range(spin_lat.N_sites):
        x, y, u = spin_lat.mps2lat_idx(i)
        assert tuple(fermion_lat.mps2lat_idx(2 * i)) == (x, y, 2 * u)
        assert tuple(fermion_lat.mps2lat_idx(2 * i + 1)) == (x, y, 2 * u + 1)


def OnsiteTermsByLatticeSite(model):
    """{(op, lattice index): strength} of the onsite terms of model"""
    onsite = {}
    for term, strength in model.all_onsite_terms().to_TermList():
        (op, i), = term
        key = (op, tuple(int(v) for v in model.lat.mps2lat_idx(i)))
        onsite[key] = onsite.get(key, 0.0) + strength
    return onsite


@pytest.mark.parametrize("model_type, flux, monopole_Q, helical",
                         [("Dirac", 0.0, 0, False), ("Dirac", 1.0, 2, False), ("Dirac", 0.0, 0, True),
                          ("Z2", 0.0, 0, False), ("Z2", 1.0, 0, False)])
def test_mean_field_couplings_are_full_cylinder_minus_removed_sites(model_type, flux, monopole_Q, helical):
    from tenpy.networks.site import FermionSite

    Lx, Ly = 6, 4
    models = {}
    for remove_sites in ("edges", None):
        spec = GutzwillerSpec(model_type, helical, remove_sites)
        params = spec.mean_field_params(flux, monopole_Q, particle_hole=True)
        params["lattice"] = spec.fermion_lattice(FermionSite("N"), Lx, Ly, True, "finite")
        models[remove_sites] = spec.mean_field_model(params)
    removed = {tuple(r) for r in models["edges"].lat.remove.tolist()}  # fermion lattice indices
    assert removed == {(0, 0, 0), (0, 0, 1), (Lx - 1, Ly - 1, 0), (Lx - 1, Ly - 1, 1)}  # both flavors of 2 spin sites

    for terms_by_site in (CouplingsByLatticeSite, OnsiteTermsByLatticeSite):
        terms = terms_by_site(models["edges"])
        full_terms = terms_by_site(models[None])
        sites_of = (lambda key: [site for _, site in key]) if terms_by_site is CouplingsByLatticeSite else \
            (lambda key: [key[1]])
        expected = {key: s for key, s in full_terms.items() if not any(site in removed for site in sites_of(key))}
        assert terms.keys() == expected.keys()
        assert all(np.isclose(terms[key], expected[key]) for key in expected)


def test_gutzwiller_dir_name():
    from Main import GutzwillerCaseDirName
    from PostProcess import TRIANGULAR_GUTZWILLER_CASE_NAME

    name = GutzwillerCaseDirName("r/", 24, 6, 1000, 0.0, "YC", "finite", 0, "Dirac", 0.0, 0, remove_sites="edges")
    assert name == "r/Dirac_finite_Lx_24_Ly_6_chi_1000_flux_0.0_YC_gsindex_0_rm_edges_monQ_0/"
    assert TRIANGULAR_GUTZWILLER_CASE_NAME.match(name[2:-1]) is not None  # the energy post-process finds it
    assert GutzwillerCaseDirName("r/", 24, 6, 1000, 0.0, "YC", "finite", 0, "Dirac", 0.0, 0, remove_sites="None") == \
        GutzwillerCaseDirName("r/", 24, 6, 1000, 0.0, "YC", "finite", 0, "Dirac", 0.0, 0)
    with pytest.raises(ValueError):
        GutzwillerCaseDirName("r/", 24, 6, 1000, 0.0, "XC", "finite", 0, "Dirac", 0.0, 0, remove_sites="edges")


def test_gutzwiller_removed_sites_finite_only(tmp_path, monkeypatch):
    import Gutzwiller

    monkeypatch.setattr(Gutzwiller, "local", False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssertionError, match="finite MPS"):
        Gutzwiller.SpinonTriangularLatticeMeanFieldGutzwillerProjection(4, "YC", "infinite", 0, "Dirac", Lx=2,
                                                                        chi_max=50, remove_sites="edges")


@pytest.mark.parametrize("model_type", ["Dirac", "Z2"])
def test_gutzwiller_run_and_energy_post_process(tmp_path, monkeypatch, model_type):
    import Gutzwiller
    from PostProcess import CalculateTriangularGutzwillerEnergy, J2_FILE, TRIANGULAR_ENERGIES_FILE
    from ResultsIO import MPS_LATTICE_FILE

    Lx, Ly = 6, 4
    monkeypatch.setattr(Gutzwiller, "local", False)  # results in the working directory
    monkeypatch.chdir(tmp_path)
    Gutzwiller.SpinonTriangularLatticeMeanFieldGutzwillerProjection(Ly, "YC", "finite", 0, model_type, Lx=Lx,
                                                                    chi_max=400, remove_sites="edges")

    psi = LoadPsi(tmp_path, "psi_gutzwiller.pkl")
    lat = LoadLattice(tmp_path, MPS_LATTICE_FILE)
    assert HasRemovedSites(lat) and psi.L == lat.N_sites == LoadLattice(tmp_path).N_sites == Lx * Ly - 2
    C = np.loadtxt(tmp_path / "spin_corr_x.csv", dtype=complex).real
    assert np.allclose(np.diag(C), 0.75)  # one spin 1/2 per site: the projection
    assert abs(np.sum(psi.expectation_value("Sz"))) < 1e-10
    # a singlet up to the Slater MPS truncation (default svd_min; the full cylinder has the same order of magnitude)
    assert 0 <= C.sum() < 0.02

    # the energy post-process on the lattice of the MPS vs the nearest neighbor bonds of the spin correlations
    (tmp_path / J2_FILE).write_text("0.0\n")
    CalculateTriangularGutzwillerEnergy(str(tmp_path) + "/")
    E_J1 = np.loadtxt(tmp_path / TRIANGULAR_ENERGIES_FILE)[1]
    bonds = [(i, j) for u1, u2, dx in lat.pairs["nearest_neighbors"]
             for i, j in zip(*lat.possible_couplings(u1, u2, dx)[:2])]
    assert np.isclose(E_J1, sum(C[i, j] for i, j in bonds) / lat.N_sites)
    plt.close("all")
