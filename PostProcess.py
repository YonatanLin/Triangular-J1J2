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
    for results_dir in lines:
        results_dir = str(results_dir)
        results_path = _normalize_results_dir(results_dir)
        psi_path = results_path / psi_filename
        if not psi_path.exists():
            raise FileNotFoundError(f"Could not find MPS file: {psi_path}")

        with open(psi_path, "rb") as f:
            psi = pickle.load(f)

        # TODO: is this the central bond always?
        central_bond = psi.L // 2
        entanglement_entropy = psi.entanglement_entropy(bonds=[central_bond])[0]
        bond_dimension = psi.chi[central_bond]

        output_path = "EE_central_bond.txt"
        np.savetxt(
            output_path,
            np.array([[central_bond, bond_dimension, entanglement_entropy]]),
            header="bond bond_dimension entanglement_entropy",
        )

        print(f"central bond: {central_bond}")
        print(f"entanglement entropy: {entanglement_entropy}")
        #print(f"saved result to: {output_path}")


def PostProcessResults(results_dirs_file, post_process_type):
    if post_process_type == ENTANGLEMENT_ENTROPY:
        CalculateCentralBondEntanglementEntropy(results_dirs_file)

    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )

PostProcessResults("postprocess_input.txt", ENTANGLEMENT_ENTROPY)