from ClusterInputConfigurations import build_parser, postprocess_input_params
from PostProcess import PostProcessResults


def main():
    args = build_parser(postprocess_input_params).parse_args()
    args_dict = vars(args)
    kwargs = {param: args_dict[param] for param, param_type in postprocess_input_params}
    PostProcessResults(**kwargs)


if __name__ == "__main__":
    main()
