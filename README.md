# Overview Cache_replacement_DRL

This is the code to show the results of the paper "Learning-Based Joint User Association and Cache Replacement in Small Cell Networks" by Sang eun jeon, Jae wook jung and Jun pyo hong.

User association and content caching in small cell networks (SCNs) are widely studied to accommodate tremendous media traffic and alleviate the bottleneck of backhaul links. SCNs are appropriate for serving users with reducing transmission latency by properly caching the contents based on request distribution. We tackle the problem with a deep reinforcement learning (DRL) that considers the content transmission process and caching of small cells to facilitatees learning.

The application uses Pytorch backend on PyCharm: JetBrains. Our paper covers several algorithms that make user association and content replacement decisions in SBS to minimize latency in wireless communication in a cacheable small cell network. I assumed a cacheable small cell network realistically. The network environment assumes a small Rayleigh fading model, and various environments can be simulated by adjusting the environment parameters. The end goal for this is to implement a reinforcement learning-based trained algorithm using a deep neural network (DNN).

### System model
<p align="center"><img src="./figure/System_model.png" width="40%" height="40%"/></p>


## Main learning in Wireless cache network
This creates a wireless network environment and creates content delivery scenarios. Using the experiences in the content delivery process, the DNN is trained through the Train function. To reduce the bias in learning, instead of proceeding with learning every episode, the experience of the episode is accumulated and repeated at a certain moment. For reference, the main_learning.py is for learning.

<p align="center"><img src="./figure/Block_diagram.png" width="40%" height="40%"/></p>

## Designing DNN model 
This is the code that designed the neural network structure. As shown in the paper, the DNN structure to be trained can be found in DNN_model.py. I went through a lot of trial and error to find the current effective DNN, and you can check the old DNN structure in the last_DNN_model folder.


## Performance test
This can confirm the performance results for the latency according to user association and cache replacement of the Deep reinforcement learning-based algorithm. The folder is composed of code for testing, and the training results of each DNN model are saved every 5000 episodes. This code can derive learning results according to three environment variables.

* Coverage (x_max, y_max)
```c
def __init__(self):
        self.x_max = 250
        self.y_max = 250
        self.BS_Location = np.array([[(-1 * self.x_max / 2.0), (self.y_max / 2.0)],
                                    [(self.x_max / 2.0), (self.y_max / 2.0)],
                                    [(-1 * self.x_max / 2.0), (-1 * self.y_max / 2.0)],
                                    [(self.x_max / 2.0), (-1 * self.y_max / 2.0)]]) # SBS location
```
* Content Popularity (Zipf's exponent)
```c
def __init__(self):
        self.alpha = 0.8
...
def Zip_funtion(self): # generate zip distribution
        m = np.sum(np.array(range(1, self.Num_file+1))**(-self.alpha))
        self.Zip_law = (np.array(range(1, self.Num_file+1))**(-self.alpha)) / m
```
* Memory (cache capacity)
```c
def __init__(self):
        self.Memory = 16
```
Modify the following code in wireless_cache_network.py. After changing, run main_testing.py.

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
