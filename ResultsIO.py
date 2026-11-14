"""
Loading the pickles of a run's results directory (the case dir): the MPS and the lattices. The cluster runs keep
the pickles in the sibling pklFiles_<case dir name>, the local runs in the case dir itself; both are looked in, so
every loader works for either (and for a pklFiles_ dir passed itself). No project imports, so that every module,
including the cluster runner scripts, can use it.
"""
import pickle
from pathlib import Path

LATTICE_FILE = "lattice.pkl"  # the lattice of the saved correlations (longer than the MPS for an iMPS)
MPS_LATTICE_FILE = "mps_lattice.pkl"  # the lattice of the MPS, saved by the Gutzwiller runs
PSI_FILENAMES = ("psi_gs.pkl", "psi_gutzwiller.pkl")  # the MPS of the DMRG and of the Gutzwiller runs


def FindPkl(results_dir, fname):
    """fname in pklFiles_<results_dir name>, else in results_dir itself (None if in neither)."""
    results_dir = Path(results_dir).expanduser()
    for d in (results_dir.parent / f"pklFiles_{results_dir.name}", results_dir):
        if (d / fname).exists():
            return d / fname
    return None


def FindPsi(results_dir, psi_fname=None):
    """The MPS file of results_dir: psi_fname, or by default the first of PSI_FILENAMES found (None if none)."""
    for fname in PSI_FILENAMES if psi_fname is None else (psi_fname,):
        path = FindPkl(results_dir, fname)
        if path is not None:
            return path
    return None


def LoadPath(path, missing_message="no such pickle"):
    """The pickle at path, found by FindPkl / FindPsi (asserts it was found)."""
    assert path is not None, missing_message
    print(f"loading {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def LoadPkl(results_dir, fname):
    """The pickle fname of results_dir (see FindPkl)."""
    return LoadPath(FindPkl(results_dir, fname), f"no {fname} for {results_dir}")


def LoadPsi(results_dir, psi_fname=None):
    """The MPS of results_dir (see FindPsi)."""
    return LoadPath(FindPsi(results_dir, psi_fname), f"no {psi_fname or PSI_FILENAMES} for {results_dir}")


def LoadLattice(results_dir, fname=LATTICE_FILE):
    """The lattice of the correlations of results_dir, or of its MPS with fname=MPS_LATTICE_FILE."""
    return LoadPkl(results_dir, fname)
