"""
    python OverlapRunner.py <dir1> <dir2> [psi1_filename] [psi2_filename]

    <psi1|psi2> for the MPS pickled in dir1 and dir2. A psi filename "auto" (or omitted) takes the first of
    PSI_FILENAMES found in the directory. For an infinite MPS the overlap is per unit cell: the dominant eigenvalue eta
    of the mixed transfer matrix (the overlap of N unit cells is eta^N); also |eta|^(1/L), per site.
    Written to dir1/overlap_with_<dir2 name>.txt. Sent to the cluster by JobUtilities/SendJobsOverlap.sub.
"""
import os
import sys
import pickle
import warnings

PSI_FILENAMES = ("psi_gutzwiller.pkl", "psi_gs.pkl")


def ResolvePsiFile(results_dir, psi_filename="auto"):
    if psi_filename != "auto":
        return os.path.join(results_dir, psi_filename)
    for fname in PSI_FILENAMES:
        if os.path.isfile(os.path.join(results_dir, fname)):
            return os.path.join(results_dir, fname)
    raise FileNotFoundError(f"none of {PSI_FILENAMES} in {results_dir}")


def LoadCanonicalPsi(psi_file):
    with open(psi_file, "rb") as f:
        psi = pickle.load(f)
    psi.canonical_form()
    return psi


def OverlapOutputFile(dir1, dir2):
    return os.path.join(dir1, f"overlap_with_{os.path.basename(os.path.normpath(dir2))}.txt")


def CalculateOverlap(dir1, dir2, psi1_filename="auto", psi2_filename="auto"):
    """ <psi1|psi2> (per unit cell for an infinite MPS), written to OverlapOutputFile(dir1, dir2). Returns it. """
    psi_files = [ResolvePsiFile(dir1, psi1_filename), ResolvePsiFile(dir2, psi2_filename)]
    psi1, psi2 = (LoadCanonicalPsi(f) for f in psi_files)
    infinite = psi1.bc == "infinite"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # tenpy's warning that the iMPS overlap is per unit cell
        eta = psi1.overlap(psi2, understood_infinite=True) if infinite else psi1.overlap(psi2)

    lines = [f"psi1: {psi_files[0]} (bc={psi1.bc}, L={psi1.L}, chi={max(psi1.chi)})",
             f"psi2: {psi_files[1]} (bc={psi2.bc}, L={psi2.L}, chi={max(psi2.chi)})",
             f"<psi1|psi2> = {complex(eta):.12f}, abs = {abs(eta):.12f}"]
    if infinite:
        lines[-1] += f", abs per site = {abs(eta) ** (1 / psi1.L):.12f} (per unit cell: dominant eigenvalue of " \
                     f"the mixed transfer matrix)"
    with open(OverlapOutputFile(dir1, dir2), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return eta


if __name__ == "__main__":
    assert len(sys.argv) in (3, 4, 5), __doc__
    CalculateOverlap(*sys.argv[1:])
