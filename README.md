# DDPG and HER for robotics

I apply DDPG+HER for a Nachi_mz07 robot arm with ROS, Gazebo and OpenAI_ROS framework
My work is the extension of the work of Tianhong Dai: 
https://github.com/TianhongDai/hindsight-experience-replay

# New feature:
I changed the output from torque value to delta position value. I want the robot to learn how to move based on position controller instead of torque controller. It's more safety in an industrial application.

I changed the HER buffer in TianhongDai/OpenAI work from a static to dynamics size, it is now applicable to the case of early ending episode

# My System
- ROS Noetic
- Python 3.8
- Pytorch 1.12.0 + Cu116
- Ubuntu 20.04.5 LTS focal
- Gym 0.21.0
- MoveIt 1

# Demo video:
- https://www.youtube.com/watch?v=6-gCsZbldZk
- https://www.youtube.com/watch?v=z-ClO1CUYug

# Instruction:
- Will be added later
