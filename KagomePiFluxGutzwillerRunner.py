from Gutzwiller import SpinonKagomeMeanFieldGutzwillerProjection
from ClusterInputConfigurations import RunFromCommandLine, kagome_gutzwiller_input_params


if __name__ == "__main__":
    RunFromCommandLine(kagome_gutzwiller_input_params, SpinonKagomeMeanFieldGutzwillerProjection)
