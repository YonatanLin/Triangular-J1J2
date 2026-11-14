import sys
from pathlib import Path

from PostProcess import SUPPORTED_POST_PROCESS_TYPES, PARENT_DIR_POST_PROCESS_TYPES


def WritePostProcessCondorCases(parent_dir, post_process_type, condor_cases_filename="postprocess_condor_cases.txt"):
    """
    Write the condor input of the post-process jobs: a line "<results_dir> <post_process_type>" for every case dir
    (containing run.log) under parent_dir, one condor job per line; for PARENT_DIR_POST_PROCESS_TYPES a single line
    with results_dir = parent_dir/, the job going over its case dirs itself. results_dir is written as
    parent_dir/<case>/ with parent_dir exactly as given, so give it relative to the cluster root (SendJobsPostProcess.sub
    uses root_dir/results_dir).
    """
    if post_process_type not in SUPPORTED_POST_PROCESS_TYPES:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}")
    parent_dir_str = str(parent_dir).replace("\\", "/").rstrip("/")
    if post_process_type in PARENT_DIR_POST_PROCESS_TYPES:  # a single job, over the case dirs of parent_dir
        results_dirs = [f"{parent_dir_str}/"]
    else:
        case_dirs = sorted(p.parent for p in Path(parent_dir).rglob("run.log"))
        results_dirs = [f"{parent_dir_str}/{case_dir.relative_to(parent_dir).as_posix()}/" for case_dir in case_dirs]
    with open(condor_cases_filename, "w") as f:
        for results_dir in results_dirs:
            f.write(f"{results_dir} {post_process_type}\n")
    print(f"wrote {len(results_dirs)} cases to {condor_cases_filename}")


if __name__ == "__main__":
    # python CreatePostProcessInput.py <parent_dir> <post_process_type> [<condor_cases_filename>]
    WritePostProcessCondorCases(*sys.argv[1:4])
