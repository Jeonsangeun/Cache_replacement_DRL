# -----Basic library-----
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
# -----local library-----
import wireless_cache_environment as cache
from conventional_method import *
import DNN_model

# -----main setting-----
max_episode = 10
coverage = 200 # Network coverage, range : [150, 300]
Zipf_ex = 0.8 # popularity exponential, range : [0.3, 2.0]
Mem = 16 # cache memory capacity range : [4, 24]
non_factor = 5 # a certain number of content popularity changes : [1, 19]
env = cache.cache_replacement(coverage, Zipf_ex, Mem, non_factor)
conventional = LFU()
latency_layer = [] # latency stack
cache_layer = [] # cache hit rate stack

# -----hyper_parameter-----
node = 400
input_size = 5 * env.F_packet + 4
output_size = 4 * env.F_packet

# -----load network parameters-----
type_DNN = 0 # 0 : FCN DQN, 1 : Propose DQN
Model_path = "CNN_c200_40000.pth" # file name

# -----select algorithm-----
algorithm = 0 # 0 : DUA-LFU, 1 : CUA-LFU, 2 : DQN-FCN & Proposed scheme

def main():

    if type_DNN == 0:
        main_DQN = DNN_model.Qnet_FCN(input_size, node, output_size) #FCN
    elif type_DNN == 1:
        main_DQN = DNN_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size) #CNN

    main_DQN.load_state_dict(torch.load(Model_path, map_location='cpu'))
    main_DQN.eval()

    conventional.init_change()

    env.Zip_funtion()  # init popularity
    request = 1000  # the number of requests

    interval = 1  # output check cycle
    cost = 0.0  # init cost
    conventional.init_change()  # start LFU

    for episode in range(max_episode):
        if episode % 500 == 0 and episode != 0:
            env.change_pop()

        conventional.init_episode()
        # ----initialization-----
        state = env.reset()
        file = env.file_request[0]
        user = env.user_location
        conventional.init_episode()

        for i in range(request * env.Num_packet):

            if algorithm == 0:
                action = conventional.CUA_LFU(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet) # CUA-LFU
            elif algorithm == 1:
                action = conventional.DUA_LFU(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet) # DUA-LFU
            elif algorithm == 2: # using main_DQN
                s = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    aa = main_DQN(s).cpu().detach().numpy()
                    Noise = np.zeros(4 * env.F_packet)
                    action = env.action_select(aa, Noise)

            next_state, reward, done, file, user = env.step(action, file, user)

            state = next_state

        cost += env.cost
        conventional.buffer_his(episode)

        if episode % interval == (interval - 1):
            latency_layer.append(cost / interval)
            print("Episode: {} cost: {}".format(episode, (cost / interval)))
            cost = 0.0

    np.save("final_acc_delay", latency_layer)

if __name__ == '__main__':
    main()