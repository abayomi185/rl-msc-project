from config import SimConfig

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import torch as th

simconfig = SimConfig()

from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

from task import NiryoOneTask
task = NiryoOneTask(name="NiryoOne", sim_config=simconfig, env=env)

log_dir = "./mip_policy7"

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[64, 32], vf=[64, 32])])

policy = CnnPolicy

total_timesteps = 5000000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="niryo_policy_checkpoint")
model = PPO(
    policy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10000,
    batch_size=500,
    learning_rate=0.00025,
    gamma=0.9995,
    device="cuda",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/niryo_policy")

env.close()