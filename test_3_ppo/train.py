from env import NiryoOneEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = "./mip_policy2"
# set headles to false to visualize training
my_env = NiryoOneEnv(headless=False)
# my_env = make_vec_env(NiryoOneEnv, n_envs=4, env_kwargs=dict(headless=False))


policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[64, 32], vf=[64, 32])])

# TODO - Policy important for observation space
# policy = CnnPolicy
policy = MultiInputPolicy

total_timesteps = 1000000

if args.test is True:
    total_timesteps = 10000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="niryo_policy_checkpoint")
model = PPO(
    policy,
    my_env,
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

my_env.close()
