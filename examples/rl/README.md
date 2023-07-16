# Reinforcement Learning Example based-on ATACOM

Implementation of Acting on the TAngent space of the Constraint Manifold (ATACOM) 
adapted to the Robot-Air-Hockey-Challenge.

## Download the pre-trained agents

``` console
python download_agents.py
```
Extract the `atacom_agents.zip` file and put into `examples/rl/agents` folder

## Run training example

``` console
python air_hockey_exp.py --env <ENV_NAME> --alg atacom-sac 
```

An example reward function is defined in `rewards.py `is.

## References 

[1] Liu, Puze, Davide Tateo, Haitham Bou Ammar, and Jan Peters. "Robot reinforcement 
learning on the constraint manifold." In Conference on Robot Learning, pp. 1357-1366. 
PMLR, 2022.

[2] Liu, Puze, Kuo Zhang, Davide Tateo, Snehal Jauhri, Zhiyuan Hu, Jan Peters, and 
Georgia Chalvatzaki. "Safe reinforcement learning of dynamic high-dimensional robotic 
tasks: navigation, manipulation, interaction." In 2023 IEEE International Conference 
on Robotics and Automation (ICRA), pp. 9449-9456. IEEE, 2023.