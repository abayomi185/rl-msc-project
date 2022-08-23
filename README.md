# Notes
Add Articulation root to the base of the robot;
See https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gui_simple_robot.html

mip_policy1 - concatenated image vertically and cnn policy - 128 x 128 image

mip_policy2 - rgb image and depth info using muti-input policy - 256 x 256 image and depth.
            - final valid training has overhead camera.

mip_policy3 - same as mip_polxicy3 with overhead camera and tweaked reward system - more rewards for being close to the cube.

runs 20-08-22 18:19 - reward function -> reward = `1 / (test_reward ** 2)`

mip_policy4 - reward function -> reward = `1 / (test_reward ** 2)`

mip_policy5 - CNNPolicy with concatenated image and depth. Reward is changed as well