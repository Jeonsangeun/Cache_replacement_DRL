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

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")

start_time = time.time()

learning_rate = 0.001
gamma = 0.99

batch_size = 4096
max_episode = 20000

pop = 5
env = cache.cache_replacement(pop)
node = 400
w_node = 1
input_size = 5 * env.F_packet + 4
output_size = 4 * env.F_packet

y_layer = []
z_layer = []

def Train(Q, Q_target, memory, optimizer):
    for i in range(30):
        state, action, reward, next_state, done = memory.sample(batch_size)

        state = state.cuda(device)
        action = action.cuda(device)
        reward = reward.cuda(device)
        next_state = next_state.cuda(device)
        done = done.cuda(device)

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
    main_DQN = Q_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size).to(device)
    target_DQN = Q_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size).to(device)

    target_DQN.load_state_dict(main_DQN.state_dict())
    target_DQN.eval()

    memory = Q_model.ReplayBuffer()

    env.Zip_funtion()
    interval = 10
    request = 1000
    cost, hit_rate = 0.0, 0.0
    optimizer = optim.Adam(main_DQN.parameters(), lr=learning_rate)
    pro = 0

    for episode in range(max_episode):

        if episode % 5000 == 0 and episode != 0:
            env.change_pop()

        state = env.reset()
        file = env.file_request[0]
        user = env.user_location
        e = max((1. / ((episode // 500) + 1)), 0.1)
        for i in range(request * env.Num_packet):

            s = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                aa = Q_model.Predict_Qnet_v6(main_DQN.eval().cpu(), s).detach().numpy()
            sigma = max(10.0 / ((episode / 200) + 1), 0.316)  # sigma^2 = 0.1
            Noise = np.random.normal(0, sigma, size=4 * env.F_packet) / 10
            action = env.action_select(aa, Noise)

            if episode % 100 == 99:
                if i == 0:
                    print(np.max(aa))
                    print(np.min(aa))

            next_state, reward, done, file, user = env.step(action, file, user)
            done_mask = 0.0 if done else 1.0

            if reward <= -500:
                reward = -500

            memory.put((state, action, reward / 100, next_state, done_mask))
            state = next_state

        cost += env.cost
        hit_rate += env.hit

        if episode % interval == (interval - 1):
            main_DQN.to(device)
            y_layer.append(cost / interval)
            z_layer.append(hit_rate / interval)

            print("Episode: {} cost: {}".format(episode, (cost / interval)))
            Train(main_DQN, target_DQN, memory, optimizer)
            target_DQN.load_state_dict(main_DQN.state_dict())
            target_DQN.eval()
            cost, hit_rate = 0.0, 0.0

        if episode % 2500 == 0 and episode != 0:
            pro += 1
            savePath = "test_model_conv0" + str(pro) + ".pth"
            torch.save(main_DQN.state_dict(), savePath)
            np.save("acc_delay", y_layer)
            np.save("cache_hit", z_layer)

    savePath = "final_model.pth"
    torch.save(main_DQN.state_dict(), savePath)

    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    np.save("final_acc_delay", y_layer)
    np.save("final_cache_hit", z_layer)

if __name__ == '__main__':
    main()