import sys
from pathlib import Path

from PostProcess import SUPPORTED_POST_PROCESS_TYPES


def WritePostProcessCondorCases(parent_dir, post_process_type, condor_cases_filename="postprocess_condor_cases.txt"):
    """
    Write the condor input of the post-process jobs: a line "<results_dir> <post_process_type>" for every case dir
    (containing run.log) under parent_dir, one condor job per line. results_dir is written as
    parent_dir/<case>/ with parent_dir exactly as given, so give it relative to the cluster root (SendJobsPostProcess.sub
    uses root_dir/results_dir).
    """
    if post_process_type not in SUPPORTED_POST_PROCESS_TYPES:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}")
    parent_dir_str = str(parent_dir).replace("\\", "/").rstrip("/")
    case_dirs = sorted(p.parent for p in Path(parent_dir).rglob("run.log"))
    with open(condor_cases_filename, "w") as f:
        for case_dir in case_dirs:
            f.write(f"{parent_dir_str}/{case_dir.relative_to(parent_dir).as_posix()}/ {post_process_type}\n")
    print(f"wrote {len(case_dirs)} cases to {condor_cases_filename}")


if __name__ == "__main__":
    # python CreatePostProcessInput.py <parent_dir> <post_process_type> [<condor_cases_filename>]
    WritePostProcessCondorCases(*sys.argv[1:4])
