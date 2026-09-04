import pickle
from pathlib import Path

import numpy as np


ENTANGLEMENT_ENTROPY = "entanglement_entropy"
SPECIAL_POINTS_STRUCTURE_FACTOR = "special_points_structure_factor"
MAGZ = "magz"
SUPPORTED_POST_PROCESS_TYPES = {ENTANGLEMENT_ENTROPY, SPECIAL_POINTS_STRUCTURE_FACTOR, MAGZ}


def _normalize_results_dir(results_dir):
    return Path(results_dir).expanduser()


def _read_results_dirs(results_dirs_filename):
    with open(results_dirs_filename, "r") as results_dirs_file:
        return [line.strip() for line in results_dirs_file.readlines() if line.strip()]


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
    return np.loadtxt(energies_file_path, dtype=np.float64)[-1]


def CalculateCentralBondEntanglementEntropy(results_dirs_filename, psi_filename="psi_gs.pkl"):
    results_dirs = _read_results_dirs(results_dirs_filename)
    output_path = "EE_central_bond.txt"
    data = np.zeros((len(results_dirs), 4))
    for ind_dir, results_dir in enumerate(results_dirs):
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
        data[ind_dir, 0] = central_bond
        data[ind_dir, 1] = bond_dimension
        data[ind_dir, 2] = entanglement_entropy
        data[ind_dir, 3] = E_gs

        print(f"central bond: {central_bond}")
        print(f"entanglement entropy: {entanglement_entropy}")
    np.savetxt(output_path, data, header="bond bond_dimension entanglement_entropy energy")


def CalculatMagz(results_dirs_filename, psi_filename="psi_gs.pkl"):
    results_dirs = _read_results_dirs(results_dirs_filename)
    for ind_dir, results_dir in enumerate(results_dirs):
        print(results_dir)
        print(psi_filename)
        psi_path = _result_file_path(results_dir, psi_filename)
        with open(psi_path, "rb") as f:
            psi = pickle.load(f)
        sz_exp = psi.expectation_value('Sz')
        np.savetxt(str(results_dir) + "sz_exp.txt", sz_exp)


def CalculateSpecialPointsStructureFactor(results_dirs_filename):
    from Main import getSpecielBzPoints
    from WaveFunctionProperties import structure_factor

    results_dirs = _read_results_dirs(results_dirs_filename)
    special_bz_points = getSpecielBzPoints()
    point_names = list(special_bz_points.keys())

    output_path = "special_points_structure_factor.txt"
    data = np.zeros((len(results_dirs) * len(point_names), 6))
    for ind_dir, results_dir in enumerate(results_dirs):
        print(results_dir)
        spin_corr_path = _result_file_path(results_dir, "spin_corr_x.csv")
        lattice_path = _result_file_path(results_dir, "lattice.pkl")

        spin_corr_x = np.loadtxt(spin_corr_path, dtype=np.complex128)
        with open(lattice_path, "rb") as f:
            lattice = pickle.load(f)

        E_gs = _load_last_energy(results_dir)
        special_points_structure_factor = np.zeros((len(point_names), 3))
        for ind_point, point_name in enumerate(point_names):
            k = special_bz_points[point_name]
            sf_at_special_point = structure_factor(spin_corr_x, lattice, k)
            special_points_structure_factor[ind_point, 0:2] = k
            special_points_structure_factor[ind_point, 2] = sf_at_special_point

            row = ind_dir * len(point_names) + ind_point
            data[row, 0] = ind_dir
            data[row, 1] = ind_point
            data[row, 2:4] = k
            data[row, 4] = sf_at_special_point
            data[row, 5] = E_gs

            print(f"{point_name}: {sf_at_special_point}")

        np.savetxt(
            _result_file_path(results_dir, "special_points_structure_factor.csv", assert_exist=False),
            special_points_structure_factor,
            header="kx ky structure_factor",
        )

    np.savetxt(
        output_path,
        data,
        header=f"dir_index point_index kx ky structure_factor energy; point_names={','.join(point_names)}",
    )



def PostProcessResults(results_dirs_file, post_process_type, description):
    if post_process_type == ENTANGLEMENT_ENTROPY:
        CalculateCentralBondEntanglementEntropy(results_dirs_file)
    elif post_process_type == SPECIAL_POINTS_STRUCTURE_FACTOR:
        CalculateSpecialPointsStructureFactor(results_dirs_file)
    elif post_process_type == MAGZ:
        CalculatMagz(results_dirs_file)
    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )

