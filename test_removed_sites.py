"""Checks of the triangular J1J2 DMRG on a finite YC cylinder with removed sites (Main.ParseRemovedSites, the odd
sector of Hu et al., PRB 92, 140403(R) (2015), Fig. 2(b)).

- "edges" removes (0, 0) and (Lx - 1, Ly - 1): one site of each edge column, exchanged by the inversion through the
  center of the cylinder; other values are refused.
- The couplings of the model are those of the full cylinder without the ones touching a removed site (compared by
  lattice index, since a removed site has no MPS index).
- The DMRG ground state energy agrees with exact diagonalization.
- The whole DMRG run (J1J2DMRG) and its post-processing (entanglement entropy, Jiang analysis) run on it.
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
