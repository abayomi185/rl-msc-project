# import NiryoOne environment
from env_alt_skrl import NiryoOneEnv
import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicModel, GaussianModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

set_seed(69)

# Define the models (stochastic and deterministic models) for the agent using helper classes 
# and programming with two approaches (layer by layer and torch.nn.Sequential class).
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        super().__init__(observation_space, action_space, device, clip_actions,
                         clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(50176, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        # print("++++Shape")
        # print(self.observation_space.shape)
        x = self.net(states.view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))

        return 1 * torch.tanh(x), self.log_std_parameter   # JetBotEnv action_space is -10 to 10


class Value(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(50176, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        # view (samples, width * height * channels) -> (samples, width, height, channels)
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        # print(self.observation_space.shape)
        return self.net(states.view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))


# Load and wrap the JetBot environment (a subclass of Gym)
env = NiryoOneEnv(headless=True)
env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=512, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models

models_ppo = {
    "policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
    "value": Value(env.observation_space, env.action_space, device)}


# Initialize the models' parameters (weights and biases) using a Gaussian distribution

for model in models_ppo.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters

cfg_ppo = PPO_DEFAULT_CONFIG.copy()

cfg_ppo["rollouts"] = 1000
cfg_ppo["learning_epochs"] = 10
cfg_ppo["mini_batches"] = 10
cfg_ppo["discount_factor"] = 0.9995
cfg_ppo["lambda"] = 0.95
cfg_ppo["policy_learning_rate"] = 0.00025
cfg_ppo["value_learning_rate"] = 0.00025
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 10
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = False
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 0.5
cfg_ppo["kl_threshold"] = 0

# logging to TensorBoard and write checkpoints each 10000 timesteps
cfg_ppo["experiment"]["write_interval"] = 10000
cfg_ppo["experiment"]["checkpoint_interval"] = 10000


agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)



# Configure and instanciate the RL trainer
cfg_trainer = {"timesteps": 500000, "headless": True, "progress_interval": 10000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.eval()

# close the environment
env.close()
