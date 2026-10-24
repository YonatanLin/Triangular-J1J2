from Main import CreateKagomeCaseDir
from ClusterInputConfigurations import (kagome_dmrg_input_params, dmrg_params_excluded_from_dirname,
                                        CreateTriangularCaseDirFromInputFile)
import sys


if __name__ == "__main__":
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], kagome_dmrg_input_params,
                                         dmrg_params_excluded_from_dirname, CreateKagomeCaseDir,
                                         "kagome_condor_cases.txt")
