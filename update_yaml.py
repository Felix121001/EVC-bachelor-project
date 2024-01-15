import yaml
import sys


def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    return True


def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def convert_if_number(s):
    if is_int(s):
        return int(s)
    elif is_number(s):
        return float(s)
    else:
        return s


def update_yaml(config_file, updates):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    for key_path, value in updates.items():
        keys = key_path.split(".")
        sub_config = config
        for key in keys[:-1]:
            sub_config = sub_config.setdefault(key, {})
        sub_config[keys[-1]] = convert_if_number(value)

    with open(config_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


if __name__ == "__main__":
    config_file = sys.argv[1]
    updates = {sys.argv[i]: sys.argv[i + 1] for i in range(2, len(sys.argv), 2)}
    update_yaml(config_file, updates)
