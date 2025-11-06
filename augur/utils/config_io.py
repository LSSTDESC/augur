import yaml
import re
import os

path_matcher = re.compile(r'.*\$\{([^}^{]+)\}.*')


def path_constructor(loader, node):
    return os.path.expandvars(node.value)


class EnvVarLoader(yaml.SafeLoader):
    pass


EnvVarLoader.add_implicit_resolver('!path', path_matcher, None)
EnvVarLoader.add_constructor('!path', path_constructor)


def parse_config(config):
    """
    Utility to parse configuration file
    """
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = f.read()
        config = yaml.load(config, Loader=EnvVarLoader)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError('config must be a dictionary or path to a config file')
    return config
