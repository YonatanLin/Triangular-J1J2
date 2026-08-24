import sys

from ClusterInputConfigurations import postprocess_input_params, CreateTriangularCaseDirFromInputFile
from PostProcess import SUPPORTED_POST_PROCESS_TYPES
from pathlib import Path

def CreatePostProcessCaseDir(main_results_dir, results_dirs_file, post_process_type, description):
    if post_process_type in SUPPORTED_POST_PROCESS_TYPES:
        dirname = main_results_dir + description + "_" + post_process_type
        Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname
    else:
        raise ValueError(
            f"Unsupported post process type: {post_process_type}. "
            f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}")


if __name__ == "__main__":
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], postprocess_input_params,
                                         [], CreatePostProcessCaseDir,
                                         "postprocess_condor_cases.txt")
