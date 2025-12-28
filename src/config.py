import os

class Config:
    def __init__(self, env_file: str = ".env"):
        self._attributes = {}
        self._load_env_file(env_file)

    def _load_env_file(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} does not exist.")

        with open(path, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                os.environ[key] = value
                self._attributes[key] = value
                self._attributes[key.lower()] = value

    def __getattr__(self, name: str):
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")
                
config = Config(env_file="../.env")