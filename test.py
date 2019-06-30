from config_parser.parser import create_config

config = create_config("default.config")

print(config.getint("train", "batch_size"))
