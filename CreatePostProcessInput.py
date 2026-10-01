import sys

from ClusterInputConfigurations import postprocess_input_params, CreateTriangularCaseDirFromInputFile
from PostProcess import SUPPORTED_POST_PROCESS_TYPES
from pathlib import Path


def CreatePostProcessCaseDir(main_results_dir, results_dir, post_process_type):
    """Each post-process job runs in (and writes its output and logs to) the results dir it processes;
    results_dir is given relative to main_results_dir."""
    if post_process_type not in SUPPORTED_POST_PROCESS_TYPES:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}")
    case_dir = main_results_dir + results_dir
    assert Path(case_dir).is_dir(), f"results dir {case_dir} does not exist"
    return case_dir


def WritePostProcessInputFile(parent_dir, post_process_type, input_filename):
    """Write a post-process input file with a row for every case dir (containing dmrg_params.json) under
    parent_dir; results_dir is written relative to parent_dir, so use parent_dir (with trailing /) as
    main_results_dir when calling this script's main."""
    parent_dir = Path(parent_dir)
    case_dirs = sorted(p.parent for p in parent_dir.rglob("dmrg_params.json"))
    with open(input_filename, "w") as f:
        f.write(" ".join(name for name, *_ in postprocess_input_params) + "\n")
        f.write("\n".join(f"{case_dir.relative_to(parent_dir).as_posix()}/ {post_process_type}"
                          for case_dir in case_dirs) + "\n")
    print(f"wrote {len(case_dirs)} cases to {input_filename}")


if __name__ == "__main__":
    # python CreatePostProcessInput.py <main_results_dir> <input_file>
    #     input_file header: results_dir post_process_type (one case per row, results_dir relative to main_results_dir)
    # python CreatePostProcessInput.py --generate <parent_dir> <post_process_type> <input_file>
    #     writes such an input file for all cases under parent_dir
    if sys.argv[1] == "--generate":
        WritePostProcessInputFile(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], postprocess_input_params,
                                             [], CreatePostProcessCaseDir,
                                             "postprocess_condor_cases.txt")
