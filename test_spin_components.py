"""Tests for the spin-component correlations (WaveFunctionProperties) and the spin-components post-process."""
import pickle

import numpy as np
import pytest
import tenpy.linalg.np_conserved as npc
from tenpy.models.lattice import Triangular
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

from WaveFunctionProperties import (CalculateSpinSpinCorrelations, CalculateSpinSpinCorrelationComponents,
                                    structure_factor)

SX = np.array([[0, 1], [1, 0]]) / 2
SY = np.array([[0, -1j], [1j, 0]]) / 2
SZ = np.diag([0.5, -0.5])


def _random_finite_mps(n_sites, seed):
    rng = np.random.default_rng(seed)
    site = SpinHalfSite(conserve=None)
    vec = rng.normal(size=2 ** n_sites) + 1j * rng.normal(size=2 ** n_sites)
    vec /= np.linalg.norm(vec)
    psi = MPS.from_full([site] * n_sites, npc.Array.from_ndarray_trivial(
        vec.reshape([2] * n_sites), labels=[f"p{i}" for i in range(n_sites)]), form="B")
    return psi, vec


def _dense_two_site(vec, n_sites, i, j, op1, op2):
    psi = vec.reshape([2] * n_sites)

    def apply(op, site, state):
        return np.moveaxis(np.tensordot(op, state, axes=(1, site)), 0, site)

    return np.vdot(vec.reshape(psi.shape), apply(op1, i, apply(op2, j, psi))) if i != j else \
        np.vdot(psi, apply(op1 @ op2, i, psi))


def test_components_match_dense_vector():
    n = 6
    psi, vec = _random_finite_mps(n, seed=0)
    comps = CalculateSpinSpinCorrelationComponents(psi, None)
    for i in range(n):
        for j in range(n):
            xx = _dense_two_site(vec, n, i, j, SX, SX)
            yy = _dense_two_site(vec, n, i, j, SY, SY)
            zz = _dense_two_site(vec, n, i, j, SZ, SZ)
            np.testing.assert_allclose(comps["longitudinal"][i, j], zz, atol=1e-10)
            np.testing.assert_allclose(comps["transverse"][i, j], xx + yy, atol=1e-10)
            np.testing.assert_allclose(comps["total"][i, j], xx + yy + zz, atol=1e-10)
            np.testing.assert_allclose(comps["xx"][i, j], 0.5 * (xx + yy), atol=1e-10)


def test_component_option_consistent_with_helper_and_alias():
    psi, _ = _random_finite_mps(4, seed=1)
    comps = CalculateSpinSpinCorrelationComponents(psi, None)
    for name in ("total", "transverse", "xx", "longitudinal"):
        np.testing.assert_allclose(CalculateSpinSpinCorrelations(psi, component=name), comps[name], atol=1e-12)
    np.testing.assert_allclose(CalculateSpinSpinCorrelations(psi, transverse_correlations=True),
                               comps["transverse"], atol=1e-12)
    np.testing.assert_allclose(CalculateSpinSpinCorrelations(psi), comps["total"], atol=1e-12)


def _write_case(tmp_path, name, psi, lat):
    """Creates tmp_path/results/<name>/ and the sibling pklFiles_<name>/ holding psi_gs.pkl and lattice.pkl."""
    case_dir = tmp_path / "results" / name
    pkl_dir = tmp_path / "results" / ("pklFiles_" + name)
    case_dir.mkdir(parents=True)
    pkl_dir.mkdir(parents=True)
    with open(pkl_dir / "psi_gs.pkl", "wb") as f:
        pickle.dump(psi, f)
    with open(pkl_dir / "lattice.pkl", "wb") as f:
        pickle.dump(lat, f)
    return case_dir


def test_postprocess_spin_components(tmp_path):
    from PostProcess import PostProcessResults, SPIN_COMPONENTS_STRUCTURE_FACTOR

    site = SpinHalfSite(conserve=None)
    lat = Triangular(3, 2, site, bc=["open", "periodic"], bc_MPS="finite")
    psi, _ = _random_finite_mps(lat.N_sites, seed=2)
    case_dir = _write_case(tmp_path, "case", psi, lat)

    PostProcessResults(str(case_dir).replace("\\", "/"), SPIN_COMPONENTS_STRUCTURE_FACTOR)  # no trailing "/"

    comps = CalculateSpinSpinCorrelationComponents(psi, lat)
    np.testing.assert_allclose(np.loadtxt(case_dir / "spin_corr_xx_x.csv"), comps["xx"].real, atol=1e-10)
    np.testing.assert_allclose(np.loadtxt(case_dir / "spin_corr_zz_x.csv"), comps["longitudinal"].real, atol=1e-10)

    sf = np.loadtxt(case_dir / "spin_components_structure_factor.csv")
    for row in sf:
        k = row[:2]
        np.testing.assert_allclose(row[2], structure_factor(comps["total"].real, lat, k), atol=1e-10)
        np.testing.assert_allclose(row[3], structure_factor(comps["xx"].real, lat, k), atol=1e-10)
        np.testing.assert_allclose(row[4], structure_factor(comps["longitudinal"].real, lat, k), atol=1e-10)

    vs_dist = np.loadtxt(case_dir / "spin_components_vs_distance.csv")
    assert vs_dist.shape == (lat.N_sites, 5)
    assert vs_dist[0, 1] == 0.0  # reference site first, at distance 0
    np.testing.assert_allclose(vs_dist[0, 2], 0.75, atol=1e-10)  # <S_i0^2> = 3/4


def test_collect_post_process_results(tmp_path):
    from PostProcess import (PostProcessResults, CollectPostProcessResults, SPIN_COMPONENTS_STRUCTURE_FACTOR,
                             ENTANGLEMENT_ENTROPY)

    site = SpinHalfSite(conserve=None)
    lat = Triangular(3, 2, site, bc=["open", "periodic"], bc_MPS="finite")
    case_dirs = []
    for ind, name in enumerate(["case_a", "case_b"]):
        psi, _ = _random_finite_mps(lat.N_sites, seed=10 + ind)
        case_dir = _write_case(tmp_path, name, psi, lat)
        np.savetxt(case_dir / "Energies.txt", [-1.0 * (ind + 1)])
        for post_process_type in (SPIN_COMPONENTS_STRUCTURE_FACTOR, ENTANGLEMENT_ENTROPY):
            PostProcessResults(str(case_dir), post_process_type)
        case_dirs.append(case_dir)

    out = tmp_path / "components.txt"
    CollectPostProcessResults(tmp_path / "results", SPIN_COMPONENTS_STRUCTURE_FACTOR, out)
    data = np.loadtxt(out)
    n_points = len(np.loadtxt(case_dirs[0] / "spin_components_structure_factor.csv"))
    assert data.shape == (2 * n_points, 8)  # dir_index point_index kx ky S_total S_xx S_zz energy
    np.testing.assert_array_equal(data[:, 0], np.repeat([0, 1], n_points))
    np.testing.assert_allclose(data[:n_points, 7], -1.0)
    np.testing.assert_allclose(data[n_points:, 7], -2.0)
    np.testing.assert_allclose(data[n_points:, 2:7], np.loadtxt(case_dirs[1] / "spin_components_structure_factor.csv"))
    dirs = out.with_name(out.name + ".dirs.txt").read_text().splitlines()
    assert dirs[0].endswith("case_a") and dirs[1].endswith("case_b")

    out_ee = tmp_path / "ee.txt"
    CollectPostProcessResults(tmp_path / "results", ENTANGLEMENT_ENTROPY, out_ee)
    ee = np.loadtxt(out_ee)
    assert ee.shape == (2, 4)
    np.testing.assert_allclose(ee[:, 3], [-1.0, -2.0])
    assert np.all(ee[:, 2] >= 0)


def test_write_post_process_condor_cases(tmp_path, monkeypatch):
    """
    A case dir is one with the condor log run.log (DMRG and Gutzwiller jobs alike; the Gutzwiller ones have no
    dmrg_params.json), found at any depth. results_dir is parent_dir/<case>/ with parent_dir exactly as given, not
    made absolute: SendJobsPostProcess.sub runs the job in root_dir/results_dir, so parent_dir is relative to the
    cluster root.
    """
    from CreatePostProcessInput import WritePostProcessCondorCases
    from PostProcess import MAGZ, VIRTUAL_PARITY_DIR_NAMES

    for name in ("case1", "geometry/case2"):
        (tmp_path / "main" / name).mkdir(parents=True)
        (tmp_path / "main" / name / "run.log").write_text("")
    (tmp_path / "main" / "not_a_case").mkdir()  # no run.log
    (tmp_path / "main" / "not_a_case" / "dmrg_params.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    cases_file = tmp_path / "postprocess_condor_cases.txt"

    for parent_dir in ("main", "main/"):  # with or without the trailing /
        WritePostProcessCondorCases(parent_dir, MAGZ)
        assert cases_file.read_text().splitlines() == [f"main/case1/ {MAGZ}", f"main/geometry/case2/ {MAGZ}"]

    absolute_parent = (tmp_path / "main").as_posix()
    WritePostProcessCondorCases(absolute_parent, MAGZ)
    assert cases_file.read_text().splitlines() == [f"{absolute_parent}/case1/ {MAGZ}",
                                                   f"{absolute_parent}/geometry/case2/ {MAGZ}"]

    # a parent dir post-process: a single job over the case dirs of parent_dir
    WritePostProcessCondorCases("main", VIRTUAL_PARITY_DIR_NAMES)
    assert cases_file.read_text().splitlines() == [f"main/ {VIRTUAL_PARITY_DIR_NAMES}"]

    with pytest.raises(ValueError):
        WritePostProcessCondorCases("main", "not_a_type")


def test_special_points_postprocess_does_not_overwrite_dmrg_output(tmp_path):
    from PostProcess import PostProcessResults, CollectPostProcessResults, SPECIAL_POINTS_STRUCTURE_FACTOR

    site = SpinHalfSite(conserve=None)
    lat = Triangular(3, 2, site, bc=["open", "periodic"], bc_MPS="finite")
    psi, _ = _random_finite_mps(lat.N_sites, seed=5)
    case_dir = _write_case(tmp_path, "case", psi, lat)
    comps = CalculateSpinSpinCorrelationComponents(psi, lat)
    np.savetxt(case_dir / "spin_corr_x.csv", comps["total"].real)
    dmrg_output = "0 1 2\n3 4 5\n"  # stands in for the headerless file written by the DMRG run
    (case_dir / "special_points_structure_factor.csv").write_text(dmrg_output)

    PostProcessResults(str(case_dir), SPECIAL_POINTS_STRUCTURE_FACTOR)
    assert (case_dir / "special_points_structure_factor.csv").read_text() == dmrg_output

    out = tmp_path / "sp.txt"
    CollectPostProcessResults(tmp_path / "results", SPECIAL_POINTS_STRUCTURE_FACTOR, out)
    data = np.loadtxt(out)
    assert data.shape[1] == 6
    first_row = np.loadtxt(case_dir / "special_points_structure_factor_postprocess.csv")[0]
    np.testing.assert_allclose(data[0, 2:5], first_row)
    assert out.read_text().splitlines()[0].endswith("point_names=M1,M2,M3,K,K_prime")


def test_spin_components_output_and_save(tmp_path):
    from WaveFunctionProperties import SpinComponentsOutput, SaveSpinComponentsOutput

    site = SpinHalfSite(conserve=None)
    lat = Triangular(3, 2, site, bc=["open", "periodic"], bc_MPS="finite")
    psi, _ = _random_finite_mps(lat.N_sites, seed=3)
    special_points = {"K": np.array([2 * np.pi / 3, 2 * np.pi / np.sqrt(3)]), "M1": np.array([0., 2 * np.pi / np.sqrt(3)])}

    out = SpinComponentsOutput(psi, lat, special_points, n1=2 * lat.Ls[0], n2=2 * lat.Ls[1])
    comps = CalculateSpinSpinCorrelationComponents(psi, lat)
    np.testing.assert_allclose(out["corr"]["total"], comps["total"].real, atol=1e-12)
    # sum rule: S_total = 2 S_xx + S_zz, on the full grid and at the special points
    assert len(out["ks"]) == (2 * 2 * lat.Ls[0]) * (2 * 2 * lat.Ls[1])
    np.testing.assert_allclose(2 * out["S_xx"] + out["S_zz"], np.array([structure_factor(comps["total"].real, lat, k)
                                                                         for k in out["ks"]]), atol=1e-10)
    sp = out["special_points"]
    np.testing.assert_allclose(2 * sp[:, 3] + sp[:, 4], sp[:, 2], atol=1e-10)

    SaveSpinComponentsOutput(tmp_path, out)
    for fname in ["spin_corr_xx_x.csv", "spin_corr_zz_x.csv", "spin_components_structure_factor.csv", "ks_xx.csv",
                  "ks_zz.csv", "spin_components_vs_distance.csv"]:
        assert (tmp_path / fname).exists(), fname
    np.testing.assert_allclose(np.loadtxt(tmp_path / "ks_xx.csv")[:, 2], out["S_xx"])
    assert (tmp_path / "spin_components_structure_factor.csv").read_text().splitlines()[0].endswith("point_names=K,M1")
