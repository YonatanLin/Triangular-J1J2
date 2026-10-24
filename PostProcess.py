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

    if psi.finite:
        PlotEntanglementProfileAndFitCentralCharge(psi, results_dir)


def PlotEntanglementProfileAndFitCentralCharge(psi, results_dir):
    """
    Quick and dirty: plots S(bond) of a finite MPS and fits the open BC CFT form
    S = (c/6) ln[(2L/pi) sin(pi x/L)] + const on the bonds between columns (x in columns, L = number of columns),
    in the central half of the columns, with separate constants for even and odd x (open BC alternation, which
    decays into the bulk; the central half suppresses its residual bias). Saves EE_profile.png and EE_profile.txt.
    """
    import matplotlib.pyplot as plt
    assert(psi.finite)
    S = np.asarray(psi.entanglement_entropy())  # bond i is between sites i and i + 1
    bonds = np.arange(len(S))
    sites_per_column = 1
    lattice_path = _result_file_path(results_dir, "lattice.pkl", assert_exist=False)
    if lattice_path.exists():
        with open(lattice_path, "rb") as f:
            sites_per_column = pickle.load(f).N_sites_per_ring
    n_columns = psi.L // sites_per_column
    column_bonds = np.arange(1, n_columns) * sites_per_column - 1  # bond after the last site of each column
    x = (column_bonds + 1) // sites_per_column
    chord = np.log((2 * n_columns / np.pi) * np.sin(np.pi * x / n_columns))
    fit = (x >= n_columns / 4) & (x <= 3 * n_columns / 4)
    if np.count_nonzero(fit) % 2:  # equally many even and odd x, an unbalanced window biases c by a few percent
        fit[np.flatnonzero(fit)[-1]] = False
    # open BC give an even/odd (in x) alternation of S: shared slope, separate constants for the two parities
    odd = (x % 2).astype(float)
    design = np.column_stack([chord, 1 - odd, odd])
    (slope, intercept_even, intercept_odd), *_ = np.linalg.lstsq(design[fit], S[column_bonds][fit], rcond=None)
    intercept = intercept_even + (intercept_odd - intercept_even) * odd  # per column bond
    c = 6 * slope
    print(f"CFT fit on {np.count_nonzero(fit)} column bonds: c = {c:.3f}, const (even, odd x) = "
          f"({intercept_even:.3f}, {intercept_odd:.3f})")

    fig, (ax_bond, ax_chord) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_bond.plot(bonds, S, ".-", color="gray", lw=0.8, label="all bonds")
    ax_bond.plot(column_bonds, S[column_bonds], "o", color="C0", label="column bonds")
    S_fit = slope * chord + intercept
    for parity in (0, 1):
        branch = fit & (x % 2 == parity)
        label = f"fit, c = {c:.2f}" if parity == 0 else None
        ax_bond.plot(column_bonds[branch], S_fit[branch], "-", color="C3", label=label)
        ax_chord.plot(chord[branch], S_fit[branch], "-", color="C3", label=label)
    ax_bond.set_xlabel("bond index")
    ax_bond.set_ylabel("S")
    ax_bond.legend()
    ax_chord.plot(chord, S[column_bonds], "o", color="C0")
    ax_chord.set_xlabel(r"$\ln[(2L/\pi)\sin(\pi x/L)]$")
    ax_chord.set_ylabel("S")
    ax_chord.legend()
    fig.tight_layout()
    fig.savefig(_result_file_path(results_dir, "EE_profile.png", assert_exist=False), bbox_inches="tight")
    plt.close(fig)
    np.savetxt(_result_file_path(results_dir, "EE_profile.txt", assert_exist=False), np.column_stack([bonds, S]),
               header=f"bond entanglement_entropy; column bond CFT fit: c={c} const_even={intercept_even} "
                      f"const_odd={intercept_odd} sites_per_column={sites_per_column}")
    return c


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
    from Main import getTriangularLatticeSpecielBzPoints
    from WaveFunctionProperties import structure_factor

    special_bz_points = getTriangularLatticeSpecielBzPoints()
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


def CalculateSpinComponentsStructureFactor(results_dir, psi_filename="psi_gs.pkl", i0=None,
                                           save_full_grid=True, n1=None, n2=None):
    """
    Calculate the transverse (<SxSx>) and longitudinal (<SzSz>) spin correlations of results_dir and their structure
    factors (see WaveFunctionProperties.SpinComponentsOutput) and save them in results_dir.
    """
    from Main import getTriangularLatticeSpecielBzPoints
    from WaveFunctionProperties import SpinComponentsOutput, SaveSpinComponentsOutput

    print(results_dir)
    psi_path = _result_file_path(results_dir, psi_filename)
    lattice_path = _result_file_path(results_dir, "lattice.pkl")
    with open(psi_path, "rb") as f:
        psi = pickle.load(f)
    with open(lattice_path, "rb") as f:
        lattice = pickle.load(f)

    # default k-grid resolution: the same as the DMRG run (Main.TriangularJ1J2DMRG)
    n1 = 2 * lattice.Ls[0] if n1 is None else n1
    n2 = 2 * lattice.Ls[1] if n2 is None else n2
    output = SpinComponentsOutput(psi, lattice, getTriangularLatticeSpecielBzPoints(), i0=i0, save_full_grid=save_full_grid,
                                  n1=n1, n2=n2)
    for point_name, row in zip(output["point_names"], output["special_points"]):
        print(f"{point_name}: total={row[2]}, xx={row[3]}, zz={row[4]}")
    SaveSpinComponentsOutput(results_dir, output)


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
