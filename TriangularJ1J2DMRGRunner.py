from Main import TriangularJ1J2DMRG
from ClusterInputConfigurations import RunFromCommandLine, dmrg_input_params


if __name__ == "__main__":
    RunFromCommandLine(dmrg_input_params, TriangularJ1J2DMRG)
