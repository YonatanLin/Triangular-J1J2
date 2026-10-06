from Gutzwiller import SpinonTriangularLatticeMeanFieldGutzwillerProjection
from ClusterInputConfigurations import build_parser, gutzwiller_input_params, input_param_name


def main():
    args = build_parser(gutzwiller_input_params).parse_args()
    args_dict = vars(args)
    kwargs = {input_param_name(param): args_dict[input_param_name(param)] for param in gutzwiller_input_params}
    SpinonTriangularLatticeMeanFieldGutzwillerProjection(**kwargs)

if __name__ == "__main__":
    main()
