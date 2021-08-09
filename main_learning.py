# Require module
import random as rd
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

# environment & conventional methods
import wireless_cache_network as cache
from conventional_method import *

# DNN model
import DNN_model
# CUDA GPU set
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

start_time = time.time()

# hyperparameter for learning
learning_rate = 0.001
gamma = 0.99
batch_size = 4096
max_episode = 10000

# environment import & DNN parameter setting
env = cache.cache_replacement()
node = 400
w_node = 1
input_size = 5 * env.F_packet + 4
output_size = 4 * env.F_packet
# data_list set
y_layer = []
z_layer = []

def Train(Q, Q_target, memory, optimizer): # training function motive by seungeunrho
    for i in range(30):
        state, action, reward, next_state, done = memory.sample(batch_size)

        state = state.cuda(device)
        action = action.cuda(device)
        reward = reward.cuda(device)
        next_state = next_state.cuda(device)
        done = done.cuda(device)

        # DDQN
        # Q_out = Q(state)
        # Q_value = Q_out.gather(1, action)
        # Q_argmax_value = Q_out.max(1)[1].unsqueeze(1)
        # Q_prime = Q_target(next_state)
        # Q_prime = Q_prime.gather(1, Q_argmax_value)

        # DQN
        Q_out = Q(state)
        Q_value = Q_out.gather(1, action)
        Q_prime = Q_target(next_state).max(1)[0].unsqueeze(1)

        target = reward + gamma * Q_prime * done
        loss = F.smooth_l1_loss(Q_value, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    # DQN_FCN model
    # main_DQN = DNN_model.Qnet_FCN(input_size, node, output_size).to(device) #FCN DQN
    # target_DQN = DNN_model.Qnet_FCN(input_size, node, output_size).to(device) #FCN DQN

    # Proposed model
    main_DQN = DNN_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size).to(device)
    target_DQN = DNN_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size).to(device)

    target_DQN.load_state_dict(main_DQN.state_dict())
    target_DQN.eval()

    memory = DNN_model.ReplayBuffer()
    env.Zip_funtion()
    # learning set & 1 episode
    interval = 10
    request = 1000
    # optimize tool
    optimizer = optim.Adam(main_DQN.parameters(), lr=learning_rate)
    # print set
    cost, hit_rate = 0.0, 0.0
    pro = 0

    for episode in range(max_episode):

        state = env.reset()
        file = env.file_request[0]
        user = env.user_location
        for i in range(request * env.Num_packet):

            #CD
            # action = CD(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet)

            #SD
            # action = SD(env.Memory, env.BS_Location, user, env.state, env.point, env.F_packet):

            # RR
            # action = env.random_action()

            # DQN
            # s = torch.from_numpy(state).float().unsqueeze(0)
            # with torch.no_grad():
            #     aa = Q_model.Predict_Qnet1(main_DQN.cpu(), s).detach().numpy()
            #     action = env.action_select(aa, Noise)

            # Proposed scheme
            s = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                aa = DNN_model.Predict_Qnet_v6(main_DQN.eval().cpu(), s).detach().numpy()
                sigma = max(10.0 / ((episode / 200) + 1), 0.316)  # sigma^2 = 0.1
                Noise = np.random.normal(0, sigma, size=4 * env.F_packet) / 10
                action = env.action_select(aa, Noise)

            # print action value
            if episode % 100 == 99:
                if i == 0:
                    print(np.max(aa))
                    print(np.min(aa))

            next_state, reward, done, file, user = env.step(action, file, user)
            done_mask = 0.0 if done else 1.0

            #
            if reward <= -500:
                reward = -500

            memory.put((state, action, reward / 20.0, next_state, done_mask))
            state = next_state
        #
        cost += env.cost
        hit_rate += env.hit

        if episode % interval == (interval - 1): # Training
            main_DQN.to(device) # GPU learning
            y_layer.append(cost / interval)
            z_layer.append(hit_rate / interval)
            print("Episode: {} cost: {} hit_rate: {}".format(episode, (cost / interval), (hit_rate / interval)))

            Train(main_DQN, target_DQN, memory, optimizer)
            target_DQN.load_state_dict(main_DQN.state_dict())
            target_DQN.eval()
            cost, hit_rate = 0.0, 0.0

        if episode % 2500 == 0 and episode != 0: # Saving learned NNs every 2500
            pro += 1
            savePath = "test_model_conv0" + str(pro) + ".pth"
            torch.save(main_DQN.state_dict(), savePath)
            np.save("acc_delay", y_layer)
            np.save("cache_hit", z_layer)

    # finished NN
    savePath = "final_model.pth"
    torch.save(main_DQN.state_dict(), savePath)

    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    np.save("final_acc_delay", y_layer)
    np.save("final_cache_hit", z_layer)

if __name__ == '__main__':
    main()
