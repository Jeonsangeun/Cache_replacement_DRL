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

### System model
<p align="center"><img src="./figure/System_model.png" width="40%" height="40%"/></p>

## Deep reinforcement algorithm

We adopted DQN as our learning algorithm, and the learning process is illustrated in the block diagram below.
<p align="center"><img src="./figure/Block_diagram.png" width="50%" height="50%"/></p>
You can check the algorithm in Training.py.

## DNN_model
This is the code that designed the neural network structure. As shown in the paper, the DNN structure to be trained can be found in DNN_model.py. I went through a lot of trial and error to find the current effective DNN, and you can check the old DNN structure in the last_DNN_model folder.

## Performance test/DQN_test
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

## Non-stationary learning 
This is a learning and testing code in a non-stationary environment where the user's popularity in the network changes over time. The popularity change depends on the pop variable in the code. When the change_pop() function occurs during an episode, the request rank of each content rises by the amount of pop. At this time, in the case of content that has no higher rank, it is lowered to the lowest rank.
```c
pop = 5
env = cache.cache_replacement(pop)
...
if episode % 500 == 0 and episode != 0:
    env.change_pop()
```
Each variable can be modified in the following part of the main code.
