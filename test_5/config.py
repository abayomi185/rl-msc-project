import yaml

with open('./cfg/config.yaml') as f:
    config = yaml.safe_load(f)

with open('./cfg/task/Niryo.yaml') as f:
    task_config = yaml.safe_load(f)

# config = {

# }

# task_config = {
#     "env": {
#         "numEnvs": 4096,
#         "envSpacing": 1.5,
#         "episodeLength": 500,
#         "startPositionNoise": 0.0,
#         "startRotationNoise": 0.0

#     }
# }

class SimConfig():
    def __init__(self):
        self._config = config
        self._cfg = task_config

    @property
    def config(self):
        return self._config
    
    @property
    def task_config(self):
        return self._cfg

# @hydra.main(config_name="config", config_path="../cfg")