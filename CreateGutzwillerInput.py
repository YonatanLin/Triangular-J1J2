import sys
from Trying2D import CreateGutzwillerCaseDir
from ClusterInputConfigurations import CreateTriangularCaseDirFromInputFile, gutzwiller_input_params


if __name__ == "__main__":
    assert(sys.argv[1][-1] == "/")
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], gutzwiller_input_params, [],
                                         CreateGutzwillerCaseDir, "gutz_condor_cases.txt")
