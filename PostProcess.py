import os
import re
from pathlib import Path

import numpy as np

from ResultsIO import FindPkl, FindPsi, LoadLattice, LoadPath, LoadPsi, LATTICE_FILE, MPS_LATTICE_FILE


ENTANGLEMENT_ENTROPY = "entanglement_entropy"
SPECIAL_POINTS_STRUCTURE_FACTOR = "special_points_structure_factor"
MAGZ = "magz"
SPIN_COMPONENTS_STRUCTURE_FACTOR = "spin_components_structure_factor"
DIMER_CORRELATIONS = "dimer_correlations"
JIANG_ANALYSIS = "jiang_analysis"
KAGOME_GUTZWILLER_ENERGY = "kagome_gutzwiller_energy"
VIRTUAL_PARITY_DIR_NAMES = "virtual_parity_dir_names"  # results_dir is the parent dir of the cases
SUPPORTED_POST_PROCESS_TYPES = {ENTANGLEMENT_ENTROPY, SPECIAL_POINTS_STRUCTURE_FACTOR, MAGZ,
                                SPIN_COMPONENTS_STRUCTURE_FACTOR, DIMER_CORRELATIONS, JIANG_ANALYSIS,
                                KAGOME_GUTZWILLER_ENERGY, VIRTUAL_PARITY_DIR_NAMES}
# run once on the parent dir of the cases (one condor job), not per case
PARENT_DIR_POST_PROCESS_TYPES = {VIRTUAL_PARITY_DIR_NAMES}

# per-case output files, which CollectPostProcessResults gathers
EE_FILE = "EE_central_bond.txt"
SPECIAL_POINTS_FILE = "special_points_structure_factor_postprocess.csv"  # not ...structure_factor.csv: the DMRG run writes that
SPIN_COMPONENTS_FILE = "spin_components_structure_factor.csv"
DIMER_CORR_FILE = "dimer_corr_x.csv"
JIANG_ENERGIES_FILE = "energies_vs_J2.txt"  # JiangAnalysis.ENERGIES_FILE (not imported: JiangAnalysis imports Main)
KAGOME_ENERGIES_FILE = "kagome_energies_vs_J2.txt"
# not KAGOME_ENERGIES_FILE: in the parent dir, it would be collected as a case
KAGOME_ENERGIES_COLLECTED_FILE = "kagome_energies_vs_J2_collected.txt"
J2_FILE = "J2.txt"  # the J2 values of the Kagome energies, as the parameter file of the overlaps runs


def _normalize_results_dir(results_dir):
    return Path(results_dir).expanduser()


def _with_trailing_slash(results_dir):
    return str(results_dir).replace("\\", "/").rstrip("/") + "/"


def _result_file_path(results_dir, filename, assert_exist=True):
    """filename in results_dir (the pickles are loaded with ResultsIO, which also looks in pklFiles_<case>)."""
    path = _normalize_results_dir(results_dir) / filename
    if assert_exist:
        assert path.exists(), f"file path {str(path)} does not exist"
    return path


def _load_last_energy(results_dir):
    energies_file_path = _result_file_path(results_dir, "Energies.txt", assert_exist=False)
    if not energies_file_path.exists():
        return np.nan
    return np.atleast_1d(np.loadtxt(energies_file_path, dtype=np.float64))[-1]


def CalculateCentralBondEntanglementEntropy(results_dir, psi_filename="psi_gs.pkl"):
    """Saves results_dir/EE_central_bond.txt: bond, bond_dimension, entanglement_entropy, energy."""
    print(results_dir)
    psi = LoadPsi(results_dir, psi_filename)

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
    sites_per_column = LoadLattice(results_dir).N_sites_per_ring if FindPkl(results_dir, LATTICE_FILE) else 1
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
    psi = LoadPsi(results_dir, psi_filename)
    sz_exp = psi.expectation_value('Sz')
    np.savetxt(str(results_dir) + "sz_exp.txt", sz_exp)


def CalculateSpecialPointsStructureFactor(results_dir):
    """Saves results_dir/special_points_structure_factor_postprocess.csv (the DMRG run owns ...structure_factor.csv): kx ky structure_factor per special BZ point."""
    from Main import getTriangularLatticeSpecielBzPoints
    from WaveFunctionProperties import structure_factor

    special_bz_points = getTriangularLatticeSpecielBzPoints()
    point_names = list(special_bz_points.keys())

    print(results_dir)
    spin_corr_x = np.loadtxt(_result_file_path(results_dir, "spin_corr_x.csv"), dtype=np.complex128)
    lattice = LoadLattice(results_dir)

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
    psi = LoadPsi(results_dir, psi_filename)
    lattice = LoadLattice(results_dir)

    # default k-grid resolution: the same as the DMRG run (Main.TriangularJ1J2DMRG)
    n1 = 2 * lattice.Ls[0] if n1 is None else n1
    n2 = 2 * lattice.Ls[1] if n2 is None else n2
    output = SpinComponentsOutput(psi, lattice, getTriangularLatticeSpecielBzPoints(), i0=i0, save_full_grid=save_full_grid,
                                  n1=n1, n2=n2)
    for point_name, row in zip(output["point_names"], output["special_points"]):
        print(f"{point_name}: total={row[2]}, xx={row[3]}, zz={row[4]}")
    SaveSpinComponentsOutput(results_dir, output)


def CalculateDimerCorrelations(results_dir, psi_filename="psi_gs.pkl"):
    """Saves results_dir/dimer_corr_x.csv: the <D_i D_j> matrix (see WaveFunctionProperties.CalculateDimerDimerCorrelations)."""
    from WaveFunctionProperties import CalculateDimerDimerCorrelations

    print(results_dir)
    psi = LoadPsi(results_dir, psi_filename)
    lattice = LoadLattice(results_dir)

    dimer_corr = CalculateDimerDimerCorrelations(psi, lattice)
    np.savetxt(_result_file_path(results_dir, DIMER_CORR_FILE, assert_exist=False), dimer_corr)


def CalculateJiangAnalysis(results_dir, psi_filename=None, n_columns=None):
    """
    The per-case analysis of Jiang et al., PRL 137, 056703 (2026) of results_dir (JiangAnalysis.AnalyzeCaseAsJiang):
    the correlations with a reference site on n_columns columns (default 4 Ly) and their sign Fourier transform, the
    nearest neighbor correlations (if the spin_components_structure_factor post-process ran), the J1-J2 energies vs
    J2 (energies_vs_J2.txt) and the Schmidt spectrum. psi_filename defaults to the DMRG or Gutzwiller MPS
    (ResultsIO.FindPsi).
    """
    from JiangAnalysis import AnalyzeCaseAsJiang

    print(results_dir)
    psi = LoadPsi(results_dir, psi_filename)
    AnalyzeCaseAsJiang(psi, results_dir, n_columns=n_columns)


def _find_J2_file(results_dir, J2_file=J2_FILE):
    """J2_file as given if it exists, else J2_file's name in results_dir, its parent dir (the dir of the cases) or
    the working directory, the first that exists."""
    J2_file = Path(J2_file)
    candidates = [J2_file] if J2_file.is_absolute() or J2_file.parent != Path(".") else []
    results_dir = _normalize_results_dir(results_dir)
    candidates += [results_dir / J2_file.name, results_dir.parent / J2_file.name, Path.cwd() / J2_file.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{J2_file} not found, looked in {[str(c) for c in candidates]}")


def _legacy_kagome_mps_lattice(psi, results_dir):
    """
    The lattice of the MPS psi of a Kagome Gutzwiller run from before the runs saved ResultsIO.MPS_LATTICE_FILE. All such
    runs are of tenpy's Kagome lattice (BuildKagomeLattice, Gutzwiller.KagomeGutzwillerSpec) with the bc of
    Main.KagomeCylinderBC; Ly is read from the case dir name, Lx off the MPS length (Main.LatticeOfMPSByRings). The
    name must have exactly the prefix of Main.KagomeGutzwillerCaseDirName of these runs, so that a run of another
    geometry (whose name would carry it) fails here rather than being evaluated on the wrong lattice.
    """
    from tenpy.networks.site import SpinHalfSite
    from Main import BuildKagomeLattice, KagomeCylinderBC, LatticeOfMPSByRings

    name = _normalize_results_dir(results_dir).name
    match = re.match(r"Kagome_(finite|infinite)_Lx_\d+_Ly_(\d+)_chi_\d+_flux_[-+0-9.e]+_gsindex_\d+(?:_|$)", name)
    assert match is not None, (f"{name} is not a Kagome Gutzwiller case dir name of the runs without a saved MPS "
                               f"lattice (Main.KagomeGutzwillerCaseDirName)")
    bc_MPS, Ly = match.group(1), int(match.group(2))
    assert psi.bc == bc_MPS, f"MPS bc {psi.bc} != bc_MPS {bc_MPS} of the case dir name"
    site = SpinHalfSite(conserve="Sz")
    return LatticeOfMPSByRings(psi, lambda Lx: BuildKagomeLattice(Lx, Ly, site, bc_MPS, bc=KagomeCylinderBC(bc_MPS)),
                               Ly)


def CalculateKagomeGutzwillerEnergy(results_dir, J2_file=J2_FILE, Delz=1.0, psi_filename="psi_gutzwiller.pkl"):
    """
    Energies per site of the Gutzwiller state of the Kagome Gutzwiller case dir results_dir (CreateKagomeGutzwillerInput)
    w.r.t. the J1J2 model on its lattice (Main.calculateGutzwillerEnergiesJ1J2), for the J2 values of J2_file (see
    _find_J2_file). The lattice is the one the run saved (ResultsIO.MPS_LATTICE_FILE), or for older runs the one rebuilt
    by _legacy_kagome_mps_lattice. Saves results_dir/kagome_energies_vs_J2.txt: J2, energy per site.
    """
    from Main import calculateGutzwillerEnergiesJ1J2

    print(results_dir)
    J2_path = _find_J2_file(results_dir, J2_file)
    J2s = np.atleast_1d(np.loadtxt(J2_path, dtype=np.float64))
    print(f"J2 values from {J2_path}: {J2s}")
    psi = LoadPsi(results_dir, psi_filename)
    if FindPkl(results_dir, MPS_LATTICE_FILE) is not None:
        lat = LoadLattice(results_dir, MPS_LATTICE_FILE)
    else:
        print(f"no {MPS_LATTICE_FILE}, rebuilding the Kagome lattice of the MPS from the case dir name")
        lat = _legacy_kagome_mps_lattice(psi, results_dir)

    energies = np.real(calculateGutzwillerEnergiesJ1J2(psi, lat, J2s, Delz))
    np.savetxt(_result_file_path(results_dir, KAGOME_ENERGIES_FILE, assert_exist=False),
               np.column_stack([J2s, energies]), header=f"J2 energy_per_site; Delz={Delz}")
    return J2s, energies


def CalculateKagomeGutzwillerEnergies(parent_dir, J2_file=J2_FILE, Delz=1.0, output_path=None):
    """
    CalculateKagomeGutzwillerEnergy of every Kagome Gutzwiller case dir (Kagome_*) directly under parent_dir (cases
    without a Gutzwiller state are skipped), then collects them into output_path (default
    parent_dir/kagome_energies_vs_J2_collected.txt, see CollectPostProcessResults). J2_file is looked for in each case dir,
    then in parent_dir and the working directory.
    """
    parent_dir = _normalize_results_dir(parent_dir)
    case_dirs = sorted(p for p in parent_dir.iterdir() if p.is_dir() and p.name.startswith("Kagome_"))
    if not case_dirs:
        raise FileNotFoundError(f"no Kagome_* case dirs in {parent_dir}")
    for case_dir in case_dirs:
        try:
            CalculateKagomeGutzwillerEnergy(_with_trailing_slash(case_dir), J2_file, Delz)
        except AssertionError as error:  # no Gutzwiller state (e.g. a failed run)
            print(f"skipping {case_dir}: {error}")
    if output_path is None:
        output_path = parent_dir / KAGOME_ENERGIES_COLLECTED_FILE
    CollectPostProcessResults(parent_dir, KAGOME_GUTZWILLER_ENERGY, output_path)


def FirstBondParities(psi):
    """The parities of the charges (2*Sz) of the first virtual bond of psi, i.e. its virtual parities."""
    leg = psi.get_B(0).get_leg("vL")
    assert leg.chinfo.qnumber == 1, f"expected a single (2*Sz) charge, got {leg.chinfo.names}"
    return set(int(q) % 2 for q in leg.to_qflat()[:, 0])


def _with_virtual_parity_tag(case_name, parity):
    """case_name with _vpar_<parity> where GutzwillerCaseNameSuffix puts it: before _bondsign_, else at the end."""
    tag = f"_vpar_{parity}"
    return case_name.replace("_bondsign_", tag + "_bondsign_", 1) if "_bondsign_" in case_name else case_name + tag


def _RetargetLinks(dirs, renames):
    """
    Points the symlinks in dirs (recursively) whose target goes through a renamed dir (a path component in renames,
    {old name: new name}) to the renamed one, e.g. a case dir's links to the files of its pklFiles_ sibling.
    """
    for d in dirs:
        if not d.exists():
            continue
        for link in [p for p in d.rglob("*") if p.is_symlink()]:
            target = os.readlink(link)
            parts = re.split(r"([/\\])", target)  # keeps the separators, so the target is rebuilt as written
            new_target = "".join(renames.get(part, part) for part in parts)
            if new_target != target:
                link.unlink()
                os.symlink(new_target, link)
                print(f"  link {link.name}: {target} -> {new_target}")


def AddVirtualParityToDirNames(parent_dir, psi_filename="psi_gutzwiller.pkl"):
    """
    Tags with _vpar_<p> the infinite Gutzwiller case dirs directly under parent_dir that were run before the
    virtual_parity param (no vpar in the name), p being the parity of the first virtual bond's charges. Its
    pklFiles_ sibling is renamed with it. A case is left as is if those parities are mixed, if the renamed dir (or
    its pklFiles_) already exists, or if it has no psi_filename. Finite cases are left untagged, as
    GutzwillerCaseDirName names them (they only have the even sector). Returns {old dir: new dir} of the renamed.
    """
    parent_dir = _normalize_results_dir(parent_dir)
    renamed = {}
    for case_dir in sorted(p for p in parent_dir.iterdir() if p.is_dir() and not p.name.startswith("pklFiles_")):
        if "vpar" in case_dir.name:
            continue
        psi_path = FindPsi(case_dir, psi_filename)
        if psi_path is None:
            print(f"skipping {case_dir.name}: no {psi_filename}")
            continue
        psi = LoadPath(psi_path)
        if psi.bc != "infinite":
            print(f"skipping {case_dir.name}: {psi.bc} MPS (untagged by convention)")
            continue
        parities = FirstBondParities(psi)
        del psi
        if len(parities) != 1:
            print(f"skipping {case_dir.name}: mixed virtual parities {sorted(parities)}")
            continue
        new_dir = case_dir.with_name(_with_virtual_parity_tag(case_dir.name, parities.pop()))
        pkl_dir, new_pkl_dir = (d.with_name(f"pklFiles_{d.name}") for d in (case_dir, new_dir))
        if new_dir.exists() or new_pkl_dir.exists():
            print(f"skipping {case_dir.name}: {new_dir.name} already exists")
            continue
        case_dir.rename(new_dir)
        if pkl_dir.exists():
            pkl_dir.rename(new_pkl_dir)
        _RetargetLinks([new_dir, new_pkl_dir], {case_dir.name: new_dir.name, pkl_dir.name: new_pkl_dir.name})
        print(f"renamed {case_dir.name} -> {new_dir.name}")
        renamed[case_dir] = new_dir
    return renamed


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
      jiang_analysis:                   energies_vs_J2.txt  (dir_index J2 energy_per_site)
      kagome_gutzwiller_energy:         kagome_energies_vs_J2_collected.txt  (dir_index J2 energy_per_site)
    The case directories are listed, in dir_index order, in <output_path>.dirs.txt.
    """
    if post_process_type == ENTANGLEMENT_ENTROPY:
        case_filename, default_output = EE_FILE, "EE_central_bond.txt"
    elif post_process_type == JIANG_ANALYSIS:
        case_filename, default_output = JIANG_ENERGIES_FILE, "energies_vs_J2.txt"
    elif post_process_type == KAGOME_GUTZWILLER_ENERGY:
        case_filename, default_output = KAGOME_ENERGIES_FILE, KAGOME_ENERGIES_COLLECTED_FILE
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
        if post_process_type in (JIANG_ANALYSIS, KAGOME_GUTZWILLER_ENERGY):
            rows.extend(np.concatenate([[ind_dir], row]) for row in data)
            header = "dir_index J2 energy_per_site"
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
    elif post_process_type == DIMER_CORRELATIONS:
        print("sarting dimer correlations calculation...")
        CalculateDimerCorrelations(results_dir)
    elif post_process_type == JIANG_ANALYSIS:
        CalculateJiangAnalysis(results_dir)
    elif post_process_type == KAGOME_GUTZWILLER_ENERGY:
        CalculateKagomeGutzwillerEnergy(results_dir)
    elif post_process_type == VIRTUAL_PARITY_DIR_NAMES:
        AddVirtualParityToDirNames(results_dir)
    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
        )
