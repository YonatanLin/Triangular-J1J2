import pickle
from pathlib import Path

import numpy as np


ENTANGLEMENT_ENTROPY = "entanglement_entropy"
SUPPORTED_POST_PROCESS_TYPES = {ENTANGLEMENT_ENTROPY}


def _normalize_results_dir(results_dir):
    return Path(results_dir).expanduser()


def CalculateCentralBondEntanglementEntropy(results_dir, psi_filename="psi_gs.pkl"):
    results_path = _normalize_results_dir(results_dir)
    psi_path = results_path / psi_filename
    if not psi_path.exists():
        raise FileNotFoundError(f"Could not find MPS file: {psi_path}")

    with open(psi_path, "rb") as f:
        psi = pickle.load(f)

    central_bond = psi.L // 2
    entanglement_entropy = psi.entanglement_entropy(bonds=[central_bond])[0]

    output_path = results_path / "entanglement_entropy_central_bond.txt"
    np.savetxt(
        output_path,
        np.array([[central_bond, entanglement_entropy]]),
        header="bond entanglement_entropy",
    )

    print(f"central bond: {central_bond}")
    print(f"entanglement entropy: {entanglement_entropy}")
    print(f"saved result to: {output_path}")
    return entanglement_entropy


def PostProcessResults(results_dir, post_process_type):
    if post_process_type == ENTANGLEMENT_ENTROPY:
        return CalculateCentralBondEntanglementEntropy(results_dir)

    raise ValueError(
        f"Unsupported post process type: {post_process_type}. "
        f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
    )
