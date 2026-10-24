import sys
from Main import CreateKagomeGutzwillerCaseDir
from ClusterInputConfigurations import CreateTriangularCaseDirFromInputFile, kagome_gutzwiller_input_params


if __name__ == "__main__":
    assert(sys.argv[1][-1] == "/")
    CreateTriangularCaseDirFromInputFile(sys.argv[1], sys.argv[2], kagome_gutzwiller_input_params, [],
                                         CreateKagomeGutzwillerCaseDir, "kagome_gutz_condor_cases.txt")
