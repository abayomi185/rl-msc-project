from env import NiryoOneEnv
from stable_baselines3 import PPO

policy_path = "./cnn_policy/niryo_policy.zip"

my_env = NiryoOneEnv(headless=False)
model = PPO.load(policy_path)

for _ in range(20):
    obs = my_env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = my_env.step(actions)
        my_env.render()

my_env.close()
