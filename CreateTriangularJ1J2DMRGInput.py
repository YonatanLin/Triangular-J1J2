from Trying2D import CreateTriangularCaseDir
from ClusterInputConfigurations import (dmrg_input_params, dmrg_params_excluded_from_dirname,
                                        CreateTriangularCaseDirFromInputFile)
import sys


if __name__ == "__main__":
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], dmrg_input_params, dmrg_params_excluded_from_dirname,
                                         CreateTriangularCaseDir)
