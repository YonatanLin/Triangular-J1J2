import pickle
from pathlib import Path

import numpy as np


ENTANGLEMENT_ENTROPY = "entanglement_entropy"
SPECIAL_POINTS_STRUCTURE_FACTOR = "special_points_structure_factor"
SUPPORTED_POST_PROCESS_TYPES = {ENTANGLEMENT_ENTROPY, SPECIAL_POINTS_STRUCTURE_FACTOR}


def _normalize_results_dir(results_dir):
    return Path(results_dir).expanduser()


def _read_results_dirs(results_dirs_filename):
    with open(results_dirs_filename, "r") as results_dirs_file:
        return [line.strip() for line in results_dirs_file.readlines() if line.strip()]


def _result_file_path(results_dir, filename):
    return _normalize_results_dir(results_dir) / filename


def _load_last_energy(results_dir):
    energies_file_path = _result_file_path(results_dir, "Energies.txt")
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
        if not psi_path.exists():
            raise FileNotFoundError(f"Could not find MPS file: {psi_path}")

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
        if not spin_corr_path.exists():
            raise FileNotFoundError(f"Could not find spin correlation file: {spin_corr_path}")
        if not lattice_path.exists():
            results_dir_split = results_dir.split("/")
            print(f"results dir split {results_dir_split}")
            results_dir_split[-2] = "pklFiles_" + results_dir_split[-2]
            results_dir = "/".join(results_dir_split)
            print(f"results dir {results_dir}")
            lattice_path = _result_file_path(results_dir, "lattice.pkl")
            if not lattice_path.exists():
                raise FileNotFoundError(f"Could not find lattice file: {lattice_path}")

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
            _result_file_path(results_dir, "special_points_structure_factor.csv"),
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

    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )

