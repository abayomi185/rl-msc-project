# Deep Reinforcement Learning for Robotics using Nvidia Omniverse Isaac Gym

### See [Full Report](https://github.com/abayomi185/link-hub/blob/main/public/MSc_Report_coao6.pdf) of this project.

## Video Presentation
https://user-images.githubusercontent.com/21103047/190185240-7de276e2-61ad-4d5f-9fd4-f08e1a60a3cf.mov

### Alternative [YouTube link](https://youtu.be/Eask4lYFeGY)

## Misc Notes
Add Articulation root to the base of the robot;
See https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gui_simple_robot.html

mip_policy1 - concatenated image vertically and cnn policy - 128 x 128 image

mip_policy2 - rgb image and depth info using muti-input policy - 256 x 256 image and depth.
            - final valid training has overhead camera.

mip_policy3 - same as mip_polxicy3 with overhead camera and tweaked reward system - more rewards for being close to the cube.

runs 20-08-22 18:19 - reward function -> reward = `1 / (test_reward ** 2)`

mip_policy4 - reward function -> reward = `1 / (test_reward ** 2)`

mip_policy5 - CNNPolicy with concatenated image and depth. Reward is changed as well

mip_policy6 - Back to basic reward from JetBot

mip_policy7 - Slight tweak to reward, included robot arm position in observation and using only red channel of image. Also continues learning from PPO2 to PPO3 @790000

mip_policy8 - CNNPolicy with tweaks to reward to introduce more penalties

mip_policy9 - SAC off-policy algorithm speculatively tends to perform better with small sample size
