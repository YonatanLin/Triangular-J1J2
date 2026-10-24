from Gutzwiller import SpinonTriangularLatticeMeanFieldGutzwillerProjection
from ClusterInputConfigurations import RunFromCommandLine, gutzwiller_input_params


if __name__ == "__main__":
    RunFromCommandLine(gutzwiller_input_params, SpinonTriangularLatticeMeanFieldGutzwillerProjection)
