import yaml
import argparse
import os


def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dict_to_namespace(d):
    """Recursively convert a nested dict (or Namespace) into an argparse.Namespace."""
    if isinstance(d, argparse.Namespace):
        d = vars(d)
    if not isinstance(d, dict):
        return d
    return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def add_arguments_from_dict(parser, config, prefix=""):
    """Recursively register all keys in a nested config dict as CLI arguments."""
    for key, value in config.items():
        arg_name = f"{prefix}{key}"
        if isinstance(value, dict):
            add_arguments_from_dict(parser, value, prefix=f"{arg_name}.")
        elif isinstance(value, bool):
            parser.add_argument(f"--{arg_name}", type=str2bool, nargs="?", const=True, default=value)
        else:
            parser.add_argument(f"--{arg_name}", type=type(value), default=value)


def nested_namespace(args):
    """Convert a flat dotted-key Namespace into a nested dict ready for dict_to_namespace."""
    out = {}
    for k, v in vars(args).items():
        parts = k.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def parse_args(default_config="configs/anatomix_config.yaml", key=None, return_dict=False):
    """Parse CLI arguments, falling back to values defined in a YAML config file.

    A --config flag can point to an alternative YAML file. Any key in the YAML can
    be overridden on the command line using dotted notation, e.g.
    --lm.sft_config.learning_rate 1e-4.

    Args:
        default_config: Path to the default config YAML relative to the repo root.
        key: If set, only the sub-dict at this key is used (e.g. "apm" for APM training).
        return_dict: If True, return a plain dict instead of a Namespace.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=default_config, help="Path to config YAML")
    config_args, _ = parser.parse_known_args()

    yaml_config = load_yaml_config(config_args.config)
    if key:
        yaml_config = yaml_config[key]

    add_arguments_from_dict(parser, yaml_config)
    args, unknowns = parser.parse_known_args()
    print(f"Ignoring unknown args: {unknowns}")

    args = nested_namespace(args)
    if not return_dict:
        args = dict_to_namespace(args)

    try:
        if args.output_dir.split("/")[-1] != args.experiment_name:
            args.output_dir = os.path.join(args.output_dir, args.experiment_name)
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception:
        pass

    return args
