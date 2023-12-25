import tomllib


with open("pyproject.toml", "rb") as f:
    VERSION = tomllib.load(f)["project"]["version"]
