import sys

from ClusterInputConfigurations import postprocess_input_params, CreateTriangularCaseDirFromInputFile
from PostProcess import SUPPORTED_POST_PROCESS_TYPES
from pathlib import Path


def CreatePostProcessCaseDir(_main_results_dir, results_dir, post_process_type):
    """Each post-process job runs in (and writes its output and logs to) the results dir it processes, so the
    condor case folder is just results_dir (relative to the cluster root, like the DMRG case dirs).
    _main_results_dir is unused: it is part of the interface shared with the DMRG/Gutzwiller input creation."""
    if post_process_type not in SUPPORTED_POST_PROCESS_TYPES:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}")
    assert Path(results_dir).is_dir(), f"results dir {results_dir} does not exist"
    return results_dir


def WritePostProcessInputFile(parent_dir, post_process_type, input_filename):
    """Write a post-process input file with a row for every case dir (containing dmrg_params.json) under
    parent_dir. results_dir is written as parent_dir/<case>/, i.e. with parent_dir exactly as given, so give it
    relative to the cluster root (where the condor input is created)."""
    parent_dir_str = str(parent_dir).replace("\\", "/").rstrip("/")
    case_dirs = sorted(p.parent for p in Path(parent_dir).rglob("dmrg_params.json"))
    with open(input_filename, "w") as f:
        f.write(" ".join(name for name, *_ in postprocess_input_params) + "\n")
        f.write("\n".join(f"{parent_dir_str}/{case_dir.relative_to(parent_dir).as_posix()}/ {post_process_type}"
                          for case_dir in case_dirs) + "\n")
    print(f"wrote {len(case_dirs)} cases to {input_filename}")


if __name__ == "__main__":
    # python CreatePostProcessInput.py <input_file>
    #     input_file header: results_dir post_process_type (one case per row)
    # python CreatePostProcessInput.py --generate <parent_dir> <post_process_type> <input_file>
    #     writes such an input file for all cases under parent_dir
    if sys.argv[1] == "--generate":
        WritePostProcessInputFile(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        CreateTriangularCaseDirFromInputFile("", sys.argv[1], postprocess_input_params, [],
                                             CreatePostProcessCaseDir, "postprocess_condor_cases.txt")
