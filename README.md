# Overview Cache_replacement_DRL

This is the code to show the results of the paper "Learning-Based Joint User Association and Cache Replacement in Small Cell Networks" by Sang eun jeon, Jae wook jung and Jun pyo hong.

User association and content caching in small cell networks (SCNs) are widely studied to accommodate tremendous media traffic and alleviate the bottleneck of backhaul links. SCNs are appropriate for serving users with reducing transmission latency by properly caching the contents based on request distribution. We tackle the problem with a deep reinforcement learning (DRL) that considers the content transmission process and caching of small cells to facilitatees learning.

The application uses Pytorch backend on PyCharm: JetBrains. Our paper covers several algorithms that make user association and content replacement decisions in SBS to minimize latency in wireless communication in a cacheable small cell network. I assumed a cacheable small cell network realistically. The network environment assumes a small Rayleigh fading model, and various environments can be simulated by adjusting the environment parameters. The end goal for this is to implement a reinforcement learning-based trained algorithm using a deep neural network (DNN).

### System model

![그림2](https://user-images.githubusercontent.com/44052428/121885036-a25bcb00-cd4e-11eb-8672-d493a7ff1022.png)


## Main learning in Wireless cache network
This creates a wireless network environment and creates content delivery scenarios. Using the experiences in the content delivery process, the DNN is trained through the Train function. To reduce the bias in learning, instead of proceeding with learning every episode, the experience of the episode is accumulated and repeated at a certain moment. For reference, the main_learning.py is for learning.

## Designing DNN model 
This is the code that designed the neural network structure. As shown in the paper, the DNN structure to be trained can be found in DNN_model.py. I went through a lot of trial and error to find the current effective DNN, and you can check the old DNN structure in the last_DNN_model folder.


## Performance test
This can confirm the performance results for the latency according to user association and cache replacement of the Deep reinforcement learning-based algorithm. The folder is composed of code for testing, and the training results of each DNN model are saved every 5000 episodes. We derived learning results according to three environmental variables.

* Coverage (x_max, y_max)
```c
def __init__(self):
        self.x_max = 215
        self.y_max = 215
```
* Content Popularity (Zipf's exponent)
```c
def __init__(self):
        self.alpha = 0.8
```
* Memory (cache capacity)
```c
def __init__(self):
        self.Memory = 16
```
Modify the following code in wireless_cache_network.py. After changing, run main_testing.py.

## Non-stationary learning 

