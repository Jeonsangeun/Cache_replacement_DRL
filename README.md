# Code in Cache_replacement_DRL
This is the code to show the results of the paper "Learning-Based Joint User Association and Cache Replacement in Small Cell Networks" by Sang eun jeon, Jae wook jung and Jun pyo hong.

The application uses Python backend on PyCharm: JetBrains. To run the code, you need the following Python library.
```c
# Require module
import torch
import numpy

# optional library
import random
import time
import matplotlib.pyplot
import collections
```

For training, the library was used Pytorch and used the version pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2. The project consists of Training.py, a learning and scenario progress code, Cache_Net_Environment.py, a network environment code, and a DNN_model.py, designed deep neural network (DNN) model. In addition, as described in the paper, test codes that can perform performance verification while changing various network environment variables were also prepared. I wanted to use an application that implements a wireless environment, but it was too complicated to configure a cache network, so I coded it myself and proceeded with the simulation.

What to pay attention to
```c
'DNN_model.py' : The neural network model code
'DQN_ _train.py' : The training main code
'wireless_cache_ _environment.py' : The network environment code
'conventional_method.py' : Comparison algorithms code such as LFU
```
The above 4 codes must exist in the same directory where I saved them for the code to work properly.

### System model
<p align="center"><img src="./figure/System_model.png" width="40%" height="40%"/></p>

## Deep reinforcement algorithm

We adopted DQN as our learning algorithm, and the learning process is illustrated in the block diagram below.
<p align="center"><img src="./figure/Block_diagram.png" width="50%" height="50%"/></p>
You can check the algorithm in Training.py.

## DNN_model
This is the code that designed the neural network structure. As shown in the paper, the DNN structure to be trained can be found in 'DNN_model.py'. For the completed neural network model '.pth', check the learning results for each environment in the performance test folder.

## Performance test/DQN_test.py
This can confirm the performance results for the latency according to user association and cache replacement of the Deep reinforcement learning-based algorithm. The performance test can be performed in the following steps. 

First, set the environment's coverage, popularity (Zipf index), and cache memory size.
```c
# -----main setting-----
coverage = 200 # Network coverage, range : [150, 300]
Zipf_ex = 0.8 # popularity exponential, range : [0.3, 2.0]
Mem = 16 # cache memory capacity range : [4, 24]
env = cache.cache_replacement(coverage, Zipf_ex, Mem)
```
And, Select the type of deep neural network. The general FCN-DQN and the proposed design scheme can be selected.
```c
# -----load network parameters-----
type_DNN = 0 # 0 : FCN DQN, 1 : Propose DQN
```
Finally, enter the file name of the loaded deep neural network.
```c
Model_path = "CNN_c200_40000.pth" # file name
```
*Optional
An algorithm is selected to simulate the existing technique.
```c
algorithm = 0 # 0 : DUA-LFU, 1 : CUA-LFU, 2 : DQN-FCN & Proposed scheme
```
Since the model of each environment has been trained, you can use the deep neural network model (.pth) in the folder of each environment.

## non-stationary environment
In the non-stationary environment directory, the training code and test code in the non-stationary environment are located.
A non-stationary environment environment is an environment in which the popularity of content changes with each episode. The popularity changes in a FIFO manner, and the extent of change is determined by the 'non_factor' variable of the main function.

```c
# -----non-stationary environment-----
non_factor = 5 # a certain number of content popularity changes : [1, 19]
env = cache.cache_replacement(coverage, Zipf_ex, Mem, non_factor)
episode_interval = 500 # Popularity changes every 500 times
```
The following parameters have been added to the basic code, and the episode process is the same.
