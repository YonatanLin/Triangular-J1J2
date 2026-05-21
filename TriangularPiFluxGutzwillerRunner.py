from Trying2D import TriangularPiFluxGutzwiller
from ClusterInputConfigurations import build_parser, gutzwiller_input_params


def main():
    args = build_parser(gutzwiller_input_params).parse_args()
    args_dict = vars(args)
    kwargs = {param: args_dict[param] for param, param_type in gutzwiller_input_params}
    TriangularPiFluxGutzwiller(**kwargs)

if __name__ == "__main__":
    main()
