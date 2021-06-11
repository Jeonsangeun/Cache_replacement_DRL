import random as rd
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wireless_cache_change_network as cache
import numpy as np
from conventional_method import *
import DNN_model
import time
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

start_time = time.time()

learning_rate = 0.001
gamma = 0.99

max_episode = 20000
pop = 5
env = cache.cache_replacement(pop)
pop_env = pop_change()
node = 400
w_node = 1

y_layer = []
z_layer = []

def main():
    pop_env.init_change()

    env.Zip_funtion()
    interval = 300
    request = 1000
    cost, hit_rate = 0.0, 0.0
    pro = 0

    for episode in range(max_episode):
        if episode % 5000 == 0 and episode != 0:
            env.change_pop()

        pop_env.init_episode()
        state = env.reset()

        file = env.file_request[0]

        user = env.user_location

        for i in range(request * env.Num_packet):

            #CD
            action = pop_env.CD_pop(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet)
            #SD
            # action = pop.SD_pop(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet)
            # RR
            # action = env.random_action()

            next_state, reward, done, file, user = env.step(action, file, user)

            state = next_state

        cost += env.cost
        hit_rate += env.hit
        pop_env.buffer_his(episode)

        if episode % interval == (interval - 1):

            y_layer.append(cost / interval)
            z_layer.append(hit_rate / interval)
            print("Episode: {} cost: {} hit_rate: {}".format(episode, (cost / interval), (hit_rate / interval)))
            cost, hit_rate = 0.0, 0.0

            print(max(pop_env.buffer_m.sum(axis=0)))

        if episode % 2500 == 0 and episode != 0:
            pro += 1
            np.save("acc_delay", y_layer)
            np.save("cache_hit", z_layer)

    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    np.save("final_acc_delay", y_layer)
    np.save("final_cache_hit", z_layer)

if __name__ == '__main__':
    main()