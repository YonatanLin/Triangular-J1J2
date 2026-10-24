from Main import KagomeJ1J2DMRG
from ClusterInputConfigurations import RunFromCommandLine, kagome_dmrg_input_params


if __name__ == "__main__":
    RunFromCommandLine(kagome_dmrg_input_params, KagomeJ1J2DMRG)
