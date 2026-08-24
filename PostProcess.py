import pickle
from pathlib import Path

import numpy as np


ENTANGLEMENT_ENTROPY = "entanglement_entropy"
SUPPORTED_POST_PROCESS_TYPES = {ENTANGLEMENT_ENTROPY}


def _normalize_results_dir(results_dir):
    return Path(results_dir).expanduser()


def CalculateCentralBondEntanglementEntropy(results_dirs_filename, psi_filename="psi_gs.pkl"):
    results_dirs_file = open(results_dirs_filename, "r")
    lines = results_dirs_file.readlines()
    output_path = "EE_central_bond.txt"
    data = np.zeros((len(lines), 4))
    for ind_dir, results_dir in enumerate(lines):
        results_dir = results_dir.split('\n')[0]
        print(results_dir)
        print(psi_filename)
        psi_path = Path(results_dir + psi_filename)
        if not psi_path.exists():
            raise FileNotFoundError(f"Could not find MPS file: {psi_path}")

        with open(psi_path, "rb") as f:
            psi = pickle.load(f)
        
        energies_file_path = results_dir + "Energies.txt"
        E_gs = np.loadtxt(energies_file_path, dtype=np.float64)[-1]
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



def PostProcessResults(results_dirs_file, post_process_type, description):
    if post_process_type == ENTANGLEMENT_ENTROPY:
        CalculateCentralBondEntanglementEntropy(results_dirs_file)

    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )

