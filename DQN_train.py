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
import wireless_cache_environment as cache
from conventional_method import *

# DNN model
import DNN_model
# CUDA GPU set
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

# -----main setting-----
max_episode = 10
coverage = 200 # Network coverage, range : [150, 300]
Zipf_ex = 0.8 # popularity exponential, range : [0.3, 2.0]
Mem = 16 # cache memory capacity range : [4, 24]
env = cache.cache_replacement(coverage, Zipf_ex, Mem)
conventional = LFU()
latency_layer = [] # latency stack
cache_layer = [] # cache hit rate stack

# -----training parameter-----
node = 400
input_size = 5 * env.F_packet + 4
output_size = 4 * env.F_packet

# -----hyperparameter for learning-----
learning_rate = 0.001
gamma = 0.99
batch_size = 4096
max_episode = 10000

# -----load network parameters-----
type_DNN = 0               # 0 : FCN DQN, 1 : Propose DQN
Model_path = "" # file name

def Train(Q, Q_target, memory, optimizer): # training function motive by seungeunrho
    for i in range(30):
        state, action, reward, next_state, done = memory.sample(batch_size)

        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        # DDQN
        # Q_outs = Q(state)
        # Next_Q_values = Q(next_state)
        # Next_Q_state_values = Q_target(next_state)

        # Q_value = Q_outs.gather(1, action)
        # Next_Q_value = Next_Q_state_values.gather(1, torch.max(Next_Q_values, 1)[1].unsqueeze(1))
        # target = reward + gamma * Next_Q_value * done

        # # DQN
        Q_out = Q(state)
        Q_value = Q_out.gather(1, action)
        Q_prime = Q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward + gamma * Q_prime * done

        loss = F.smooth_l1_loss(Q_value, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env.Zip_funtion() # init popularity
    request = 1000 # the number of requests
    interval = 20 # output check cycle
    cost, hit_rate, tr_epoch = 0.0, 0.0, 0 # init cost
    max_episode = 20000

    for episode in range(max_episode):

        state = env.reset()
        file = env.file_request[0]
        user = env.user_location
        for i in range(request * env.Num_packet):

            # Proposed scheme
            aa = 0
            if type_DNN == 0:
                s = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    aa = net.Predict_Qnet_FCN(main_DQN.eval().cpu(), s).detach().numpy()
            elif type_DNN == 1:
                s = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    aa = net.Predict_Qnet_v6(main_DQN.eval().cpu(), s).detach().numpy()

            sigma = max(10.0 / ((episode / 200) + 1), 0.316)  # sigma^2 = 0.1
            Noise = np.random.normal(0, sigma, size=4 * env.F_packet) / 10
            action = env.action_select(aa, Noise)

            next_state, reward, done, file, user = env.step(action, file, user)
            done_mask = 0.0 if done else 1.0

            # Avoid overestimate
            if reward <= -500:
                reward = -500

            memory.put((state, action, reward / 20.0, next_state, done_mask))
            state = next_state

        # Validation_Q-value
        if episode % 100 == 99:
            print(np.max(aa))
            print(np.min(aa))
        # Accumulate
        cost += env.cost
        hit_rate += env.hit

        if episode % interval == (interval - 1): # Training
            # main_DQN.to(device) # GPU learning
            latency_layer.append(cost / interval)
            cache_layer.append(hit_rate / interval)
            print("Episode: {} cost: {} hit_rate: {}".format(episode, (cost / interval), (hit_rate / interval)))
            Train(main_DQN, target_DQN, memory, optimizer)
            target_DQN.load_state_dict(main_DQN.state_dict())
            target_DQN.eval()
            cost, hit_rate = 0.0, 0.0

        if episode % 2500 == 0 and episode != 0: # Saving learned NNs every 2500 (check-point)
            tr_epoch += 1
            savePath = "Result/test_model_conv" + str(tr_epoch) + ".pth"
            torch.save(main_DQN.state_dict(), savePath)
            np.save("Result/acc_delay", latency_layer)
            np.save("Result/cache_hit", cache_layer)

    # finished NN
    savePath = "Result/final_model.pth"
    torch.save(main_DQN.state_dict(), savePath)

    np.save("Result/final_acc_delay", latency_layer)
    np.save("Result/final_cache_hit", cache_layer)

if __name__ == '__main__':
    main()
