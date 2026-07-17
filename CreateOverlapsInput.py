import sys
from Trying2D import CreateOverlapsCaseDir
from ClusterInputConfigurations import CreateTriangularCaseDirFromInputFile, dmrg_gutz_comp_input_params


if __name__ == "__main__":
    assert(sys.argv[1][-1] == "/")
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], dmrg_gutz_comp_input_params,
                                         [], CreateOverlapsCaseDir,
                                         "overlaps_condor_cases.txt")
