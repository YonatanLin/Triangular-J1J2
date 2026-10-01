import pickle
from pathlib import Path

import numpy as np


ENTANGLEMENT_ENTROPY = "entanglement_entropy"
SPECIAL_POINTS_STRUCTURE_FACTOR = "special_points_structure_factor"
MAGZ = "magz"
SPIN_COMPONENTS_STRUCTURE_FACTOR = "spin_components_structure_factor"
SUPPORTED_POST_PROCESS_TYPES = {ENTANGLEMENT_ENTROPY, SPECIAL_POINTS_STRUCTURE_FACTOR, MAGZ,
                                SPIN_COMPONENTS_STRUCTURE_FACTOR}

# per-case output files, which CollectPostProcessResults gathers
EE_FILE = "EE_central_bond.txt"
SPECIAL_POINTS_FILE = "special_points_structure_factor_postprocess.csv"  # not ...structure_factor.csv: the DMRG run writes that
SPIN_COMPONENTS_FILE = "spin_components_structure_factor.csv"


def _normalize_results_dir(results_dir):
    return Path(results_dir).expanduser()


def _with_trailing_slash(results_dir):
    # _result_file_path locates the pklFiles_<case> directory by splitting the path on "/"
    return str(results_dir).replace("\\", "/").rstrip("/") + "/"


def _result_file_path(results_dir, filename, assert_exist=True):
    default_path = _normalize_results_dir(results_dir) / filename

    pkl_results_dir_split = results_dir.split("/")
    pkl_results_dir_split[-2] = "pklFiles_" + pkl_results_dir_split[-2]
    pkl_results_dir = "/".join(pkl_results_dir_split)
    pkl_path = _normalize_results_dir(pkl_results_dir) / filename

    default_path_exists = default_path.exists()
    pkl_path_exists = pkl_path.exists()
    if filename.split(".")[-1] == "pkl":
        if assert_exist:
            assert pkl_path_exists, f"pkl file path {str(pkl_path)} does not exist"
        return pkl_path
    else:
        if assert_exist:
            assert default_path_exists, f"file path {str(default_path)} does not exist"
        return default_path



def _load_last_energy(results_dir):
    energies_file_path = _result_file_path(results_dir, "Energies.txt", assert_exist=False)
    if not energies_file_path.exists():
        return np.nan
    return np.atleast_1d(np.loadtxt(energies_file_path, dtype=np.float64))[-1]


def CalculateCentralBondEntanglementEntropy(results_dir, psi_filename="psi_gs.pkl"):
    """Saves results_dir/EE_central_bond.txt: bond, bond_dimension, entanglement_entropy, energy."""
    print(results_dir)
    print(psi_filename)
    psi_path = _result_file_path(results_dir, psi_filename)
    with open(psi_path, "rb") as f:
        psi = pickle.load(f)

    E_gs = _load_last_energy(results_dir)
    # TODO: is this the central bond always?
    central_bond = psi.L // 2 - 1
    entanglement_entropy = psi.entanglement_entropy(bonds=[central_bond])[0]
    bond_dimension = psi.chi[central_bond]
    print(f"central bond: {central_bond}")
    print(f"entanglement entropy: {entanglement_entropy}")
    np.savetxt(_result_file_path(results_dir, EE_FILE, assert_exist=False),
               np.array([[central_bond, bond_dimension, entanglement_entropy, E_gs]]),
               header="bond bond_dimension entanglement_entropy energy")


def CalculatMagz(results_dir, psi_filename="psi_gs.pkl"):
    print(results_dir)
    print(psi_filename)
    psi_path = _result_file_path(results_dir, psi_filename)
    with open(psi_path, "rb") as f:
        psi = pickle.load(f)
    sz_exp = psi.expectation_value('Sz')
    np.savetxt(str(results_dir) + "sz_exp.txt", sz_exp)


def CalculateSpecialPointsStructureFactor(results_dir):
    """Saves results_dir/special_points_structure_factor_postprocess.csv (the DMRG run owns ...structure_factor.csv): kx ky structure_factor per special BZ point."""
    from Main import getSpecielBzPoints
    from WaveFunctionProperties import structure_factor

    special_bz_points = getSpecielBzPoints()
    point_names = list(special_bz_points.keys())

    print(results_dir)
    spin_corr_path = _result_file_path(results_dir, "spin_corr_x.csv")
    lattice_path = _result_file_path(results_dir, "lattice.pkl")

    spin_corr_x = np.loadtxt(spin_corr_path, dtype=np.complex128)
    with open(lattice_path, "rb") as f:
        lattice = pickle.load(f)

    special_points_structure_factor = np.zeros((len(point_names), 3))
    for ind_point, point_name in enumerate(point_names):
        k = special_bz_points[point_name]
        sf_at_special_point = structure_factor(spin_corr_x, lattice, k)
        special_points_structure_factor[ind_point, 0:2] = k
        special_points_structure_factor[ind_point, 2] = sf_at_special_point
        print(f"{point_name}: {sf_at_special_point}")

    np.savetxt(
        _result_file_path(results_dir, SPECIAL_POINTS_FILE, assert_exist=False),
        special_points_structure_factor,
        header=f"kx ky structure_factor; point_names={','.join(point_names)}",
    )


def _reference_site_index(lattice, n_sites):
    """MPS index of the site closest to the geometric center of the first n_sites sites."""
    positions = _site_positions(lattice, n_sites)
    return int(np.argmin(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))


def _site_positions(lattice, n_sites):
    return np.array([lattice.position(lattice.mps2lat_idx(i)) for i in range(n_sites)])


def CalculateSpinComponentsStructureFactor(results_dir, psi_filename="psi_gs.pkl", i0=None,
                                           save_full_grid=True, n1=None, n2=None):
    """
    Calculate the transverse (<SxSx> = 1/4(<S+S-> + <S-S+>), using U(1) symmetry) and longitudinal (<SzSz>)
    spin correlations of results_dir, and their structure factors, as in Fig. 7 of Gallegos et al.,
    PRL 134, 196702 (2025).

    Saves in results_dir: spin_corr_{xx,zz}_x.csv (real-space matrices), spin_components_structure_factor.csv
    (special BZ points), spin_components_vs_distance.csv (<S_i0.S_j>, <SxSx>, <SzSz> vs distance from site i0;
    default i0 = site closest to the lattice center) and optionally the full-grid S(q) of each component.
    """
    from Main import getSpecielBzPoints
    from WaveFunctionProperties import (CalculateSpinSpinCorrelationComponents, structure_factor,
                                        ComputeMomentumSpaceStructureFactor)

    special_bz_points = getSpecielBzPoints()
    point_names = list(special_bz_points.keys())
    components = ("total", "xx", "longitudinal")

    print(results_dir)
    psi_path = _result_file_path(results_dir, psi_filename)
    lattice_path = _result_file_path(results_dir, "lattice.pkl")
    with open(psi_path, "rb") as f:
        psi = pickle.load(f)
    with open(lattice_path, "rb") as f:
        lattice = pickle.load(f)

    corrs = CalculateSpinSpinCorrelationComponents(psi, lattice)
    corrs = {name: np.real_if_close(corrs[name], tol=1e6) for name in components}
    for name, fname in (("xx", "spin_corr_xx_x.csv"), ("longitudinal", "spin_corr_zz_x.csv")):
        np.savetxt(_result_file_path(results_dir, fname, assert_exist=False), corrs[name])

    # structure factor at the special BZ points
    points_data = np.zeros((len(point_names), 5))
    for ind_point, point_name in enumerate(point_names):
        k = special_bz_points[point_name]
        sf = [structure_factor(corrs[name], lattice, k) for name in components]
        points_data[ind_point] = [k[0], k[1], *sf]
        print(f"{point_name}: total={sf[0]}, xx={sf[1]}, zz={sf[2]}")
    np.savetxt(_result_file_path(results_dir, SPIN_COMPONENTS_FILE, assert_exist=False),
               points_data, header="kx ky S_total S_xx S_zz; point_names=" + ",".join(point_names))

    # full-grid structure factors
    if save_full_grid:
        for name, fname in (("xx", "ks_xx.csv"), ("longitudinal", "ks_zz.csv")):
            ks, Sk = ComputeMomentumSpaceStructureFactor(corrs[name], lattice, n1=n1, n2=n2)
            np.savetxt(_result_file_path(results_dir, fname, assert_exist=False),
                       np.column_stack([ks, Sk]), header="kx ky S(k)")

    # real-space correlations vs distance from the reference site (Fig. 7(b))
    n_sites = corrs["total"].shape[0]
    ref = _reference_site_index(lattice, n_sites) if i0 is None else i0
    positions = _site_positions(lattice, n_sites)
    distances = np.linalg.norm(positions - positions[ref], axis=1)
    order = np.argsort(distances, kind="stable")
    vs_distance = np.column_stack([order, distances[order], corrs["total"][ref, order],
                                   corrs["xx"][ref, order], corrs["longitudinal"][ref, order]])
    np.savetxt(_result_file_path(results_dir, "spin_components_vs_distance.csv", assert_exist=False),
               vs_distance, header=f"reference_site={ref}; columns: site distance S_total S_xx S_zz")


# ----------------------------------------------------------------------
# Collecting the per-case outputs of many results dirs (run locally after the cluster jobs finished)
# ----------------------------------------------------------------------
def _find_case_dirs(parent_dir, case_filename):
    """All directories under parent_dir (or parent_dir itself) containing case_filename, sorted."""
    return sorted({path.parent for path in Path(parent_dir).rglob(case_filename)})


def _header_point_names(path):
    with open(path) as f:
        return f.readline().split("point_names=")[-1].strip().split(",")


def CollectPostProcessResults(parent_dir, post_process_type, output_path=None):
    """
    Collect the per-case output files of all case directories under parent_dir into one summary file, in the
    format the serial post-process used to write:
      entanglement_entropy:             EE_central_bond.txt  (bond bond_dimension entanglement_entropy energy)
      special_points_structure_factor:  special_points_structure_factor.txt
      spin_components_structure_factor: special_points_structure_factor_components.txt
    The case directories are listed, in dir_index order, in <output_path>.dirs.txt.
    """
    if post_process_type == ENTANGLEMENT_ENTROPY:
        case_filename, default_output = EE_FILE, "EE_central_bond.txt"
    elif post_process_type == SPECIAL_POINTS_STRUCTURE_FACTOR:
        case_filename, default_output = SPECIAL_POINTS_FILE, "special_points_structure_factor.txt"
    elif post_process_type == SPIN_COMPONENTS_STRUCTURE_FACTOR:
        case_filename, default_output = SPIN_COMPONENTS_FILE, "special_points_structure_factor_components.txt"
    else:
        raise ValueError(f"Nothing to collect for post process type {post_process_type}")
    output_path = default_output if output_path is None else str(output_path)

    case_dirs = _find_case_dirs(parent_dir, case_filename)
    if not case_dirs:
        raise FileNotFoundError(f"no {case_filename} found under {parent_dir}")

    rows = []
    header = None
    for ind_dir, case_dir in enumerate(case_dirs):
        case_path = case_dir / case_filename
        E_gs = _load_last_energy(_with_trailing_slash(case_dir))
        data = np.loadtxt(case_path, ndmin=2)
        if post_process_type == ENTANGLEMENT_ENTROPY:
            rows.append(data[0])
            header = "bond bond_dimension entanglement_entropy energy"
            continue
        names = _header_point_names(case_path)
        for ind_point in range(data.shape[0]):
            rows.append(np.concatenate([[ind_dir, ind_point], data[ind_point], [E_gs]]))
        if post_process_type == SPECIAL_POINTS_STRUCTURE_FACTOR:
            header = f"dir_index point_index kx ky structure_factor energy; point_names={','.join(names)}"
        else:
            header = f"dir_index point_index kx ky S_total S_xx S_zz energy; point_names={','.join(names)}"

    np.savetxt(output_path, np.array(rows), header=header)
    with open(output_path + ".dirs.txt", "w") as f:
        f.write("\n".join(f"{ind} {case_dir}" for ind, case_dir in enumerate(case_dirs)) + "\n")
    print(f"collected {len(case_dirs)} cases into {output_path}")


def PostProcessResults(results_dir, post_process_type):
    results_dir = _with_trailing_slash(results_dir)
    if post_process_type == SPIN_COMPONENTS_STRUCTURE_FACTOR:
        CalculateSpinComponentsStructureFactor(results_dir)
    elif post_process_type == ENTANGLEMENT_ENTROPY:
        CalculateCentralBondEntanglementEntropy(results_dir)
    elif post_process_type == SPECIAL_POINTS_STRUCTURE_FACTOR:
        CalculateSpecialPointsStructureFactor(results_dir)
    elif post_process_type == MAGZ:
        CalculatMagz(results_dir)
    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )
