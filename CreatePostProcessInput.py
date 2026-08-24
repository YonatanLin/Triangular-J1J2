import sys

from ClusterInputConfigurations import postprocess_input_params, CreateTriangularCaseDirFromInputFile
from PostProcess import SUPPORTED_POST_PROCESS_TYPES


def CreatePostProcessCaseDir(results_dirs_file, post_process_type, description):
    if post_process_type == SUPPORTED_POST_PROCESS_TYPES[0]:
        return description + "_" + SUPPORTED_POST_PROCESS_TYPES[0]
    raise ValueError(
        f"Unsupported post process type: {post_process_type}. "
        f"Supported types: {sorted(SUPPORTED_POST_PROCESS_TYPES)}"
    )


if __name__ == "__main__":
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], postprocess_input_params,
                                         [], CreatePostProcessCaseDir,
                                         "condor_cases.txt")