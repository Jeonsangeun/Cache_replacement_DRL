import random as rd
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wireless_cache_network as cache
import numpy as np
from conventional_method import *
import DNN_model
import time
import matplotlib.pyplot as plt
from celluloid import Camera

import matplotlib as mpl
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)

max_episode = 10
env = cache.cache_replacement()
node = 400
w_node = 1
input_size = 5 * env.F_packet + 4
output_size = 4 * env.F_packet
y_layer = []

def main():

    # main_DQN = DNN_model.Qnet_FCN(input_size, node, output_size) #FCN
    main_DQN = DNN_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size) #CNN
    main_DQN.load_state_dict(torch.load(".pth", map_location='cpu'))
    main_DQN.eval()

    # view parameter
    # print("main_DQN:", main_DQN)
    # print("main_DQN1 value:", list(main_DQN.Lc.parameters())

    state_1 = np.zeros([env.F_packet])
    state_2 = np.zeros([env.F_packet])
    state_3 = np.zeros([env.F_packet])
    state_4 = np.zeros([env.F_packet])

    env.Zip_funtion()
    interval = 1
    request = 1000
    cost = 0.0
    start_time = time.time()

    for episode in range(max_episode):
        state = env.reset()
        file = env.file_request[0]
        user = env.user_location
        for i in range(request * env.Num_packet):
            # cache_state
            state_1 += env.state[0]
            state_2 += env.state[1]
            state_3 += env.state[2]
            state_4 += env.state[3]

            # CD
            # action = CD(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet)
            # SD
            # action = SD(env.Memory, env.BS_Location, user, env.state, env.point, env.F_packet)
            # No cache
            # action = NO(env.Memory, env.BS_Location, user, env.state, env.point, env.F_packet)

            # CNN
            s = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                aa = main_DQN(s).cpu().detach().numpy()
                Noise = np.zeros(4 * env.F_packet)
                action = env.action_select(aa, Noise)

            next_state, reward, done, file, user = env.step(action, file, user)

            # If you want to see the environment status by time step, remove the comment.
            # print("----------------------------------")
            # print("file", file)
            # print("user", user)
            # print("action_sbs", action // env.F_packet)
            # print("action_rep", action % env.F_packet)
            # print("state")
            # env.print()
            # print("reward", reward)
            # print("----------------------------------")

            state = next_state
            
        cost += env.cost

        if episode % interval == (interval - 1):
            y_layer.append(cost / interval)
            print("Episode: {} cost: {}".format(episode, (cost / interval)))
            cost = 0.0
            print(env.MS_error)

    state_1 = state_1 / di
    state_2 = state_2 / di
    state_3 = state_3 / di
    state_4 = state_4 / di

    two_bottom = np.add(state_1, state_2)
    three_bottom = np.add(two_bottom, state_3)

    state_all = np.array([state_1, state_2, state_3, state_4])

    x = range(env.F_packet)
    x_1 = range(0, env.F_packet+1, env.Num_packet)
    x_2 = ["{:0^d}".format(x) for x in np.arange(0, env.F_packet + 1, env.Num_packet)]
    p1 = plt.bar(x, state_1, color='b', hatch='/')
    plt.xticks(x_1, x_2, rotation=-30)
    p2 = plt.bar(x, state_2, color='r', hatch='\\', bottom=state_1)
    plt.xticks(x_1, x_2, rotation=-30)
    p3 = plt.bar(x, state_3, color='g', hatch='-', bottom=two_bottom)
    plt.xticks(x_1, x_2, rotation=-30)
    p4 = plt.bar(x, state_4, color='silver', hatch='o', bottom=three_bottom)
    plt.xticks(x_1, x_2, rotation=-30)
    plt.ylabel("Sum cache rate of 4 SBSs", fontsize=10)
    plt.xlabel("index $\it{j}$ chunks", fontsize=11)

    plt.legend((p4[0], p3[0], p2[0], p1[0]), ("Small-cell BS4", "Small-cell BS3", "Small-cell BS2",  "Small-cell BS1"))
    plt.ylim(0, 4.0)
    plt.show()

    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
