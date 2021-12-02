-----Basic library-----
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
-----local library-----
import wireless_cache_environment as cache
from conventional_method import *
import DNN_model

-----main setting-----
max_episode = 10
coverage = 200 # Network coverage, range : [150, 300]
Zipf_ex = 0.8 # popularity exponential, range : [0.3, 2.0]
Mem = 16 # cache memory capacity range : [4, 24]
env = cache.cache_replacement(coverage, Zipf_ex, Mem)
conventional = LFU()
y_layer = []

-----training parameter-----
node = 400
input_size = 5 * env.F_packet + 4
output_size = 4 * env.F_packet

-----load network parameters-----
type_DNN = 0 # 0 : FCN DQN, 1 : Propose DQN
Model_path = "CNN_c200_40000.pth" # file name

-----select algorithm-----
algorithm = 0 # 0 : DUA-LFU, 1 : CUA-LFU, 2 : DQN-FCN & Proposed scheme

def main():
    
    if type_DNN == 0:
        main_DQN = DNN_model.Qnet_FCN(input_size, node, output_size) #FCN
    elif type_DNN == 1:
        main_DQN = DNN_model.Qnet_v6(env.Num_packet, env.Num_file, env.F_packet, node, output_size) #CNN

    main_DQN.load_state_dict(torch.load(Model_path, map_location='cpu'))
    main_DQN.eval()

    # If you want to view the parameter please remove below comments
    # print("main_DQN:", main_DQN)
    # print("main_DQN1 value:", list(main_DQN.Lc.parameters())

    # To make cache probability
    state_1 = np.zeros([env.F_packet])
    state_2 = np.zeros([env.F_packet])
    state_3 = np.zeros([env.F_packet])
    state_4 = np.zeros([env.F_packet])
    
    env.Zip_funtion() # init popularity
    request = 1000 # the number of requests
    
    interval = 1 # output check cycle
    cost = 0.0 # init cost
    conventional.init_change() # start LFU
    
    for episode in range(max_episode):
        ----initialization-----
        state = env.reset()
        file = env.file_request[0]
        user = env.user_location
        conventional.init_episode()
        
        for i in range(request * env.Num_packet):
            # cache_state
            state_1 += env.state[0]
            state_2 += env.state[1]
            state_3 += env.state[2]
            state_4 += env.state[3]
            
            if algorithm == 0:
                action = conventional.CD_pop(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet) # CUA-LFU
            elif algorithm == 1:
                action = action = conventional.SD_pop(env.Memory, env.BS_Location, user, env.state, env.point, file, env.F_packet) # DUA-LFU
            elif algorithm == 2: # using main_DQN
                s = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    aa = main_DQN(s).cpu().detach().numpy()
                    Noise = np.zeros(4 * env.F_packet)
                    action = env.action_select(aa, Noise)

            next_state, reward, done, file, user = env.step(action, file, user)

            # If you want to see the environment status by time step, remove below comments
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
        conventional.buffer_his(episode)
        
        if episode % interval == (interval - 1):
            y_layer.append(cost / interval)
            print("Episode: {} cost: {}".format(episode, (cost / interval)))
            cost = 0.0
            
    di = 10*request * env.Num_packet
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

if __name__ == '__main__':
    main()
