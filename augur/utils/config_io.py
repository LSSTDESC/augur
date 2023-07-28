def parse_config(config):
    """
    Utility to parse configuration file
    """
    if isinstance(config, str):
        import yaml
        with open(config) as f:
            config = yaml.safe_load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError('config must be a dictionary or path to a config file')
    return config
