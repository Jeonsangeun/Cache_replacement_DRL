import numpy as np

def Distance(x, y):  # 좌표간의 거리
    return np.sqrt(np.sum((x - y) ** 2))

def CD(Memory, BS_Location, user, state,point, file, F_packet):
    action = -1
    aa = np.zeros([4, Memory])
    cc = np.zeros([4])
    dis1 = 1000
    dis2 = 1000
    cc[0] = Distance(BS_Location[0], user)
    cc[1] = Distance(BS_Location[1], user)
    cc[2] = Distance(BS_Location[2], user)
    cc[3] = Distance(BS_Location[3], user)

    aa[0] = np.where(state[0] == point)[0]
    aa[1] = np.where(state[1] == point)[0]
    aa[2] = np.where(state[2] == point)[0]
    aa[3] = np.where(state[3] == point)[0]
    for i in range(4):
        if file in aa[i]:
            if dis1 > cc[i]:
                dis1 = cc[i]
                action = F_packet * i + np.max(aa[i])
    if action == -1:
        for i in range(4):
            if dis2 > cc[i]:
                dis2 = cc[i]
                action = F_packet * i + np.max(aa[i])

    return int(action)

def CD_packet(Memory, BS_Location, user, state, point, file_reauest, F_packet, able_key):
    action = -1
    state = state.reshape([1, 4 * F_packet])
    can_action = able_key * state
    can_action = can_action.reshape([4, F_packet])

    file_in = np.where(can_action == point)

    aa = np.zeros([4, Memory])
    cc = np.zeros([4])

    dis1 = 1000

    cc[0] = Distance(BS_Location[0], user)
    cc[1] = Distance(BS_Location[1], user)
    cc[2] = Distance(BS_Location[2], user)
    cc[3] = Distance(BS_Location[3], user)

    if file_reauest in file_in[1]:
        pp = np.where(file_in[1] == file_reauest)[0]
        for k in pp:
          if dis1 > cc[file_in[0][k]]:
              dis1 = cc[file_in[0][k]]
              action = (F_packet * file_in[0][k]) + file_in[1][k]
    else:
        dis2 = np.argmin(cc)
        kk = np.where(file_in[0] == dis2)[0]
        action = F_packet*dis2 + max(file_in[1][kk])

    return int(action)

def SD(Memory, BS_Location, user, state, point, F_packet):
    action = -1
    aa = np.zeros([4, Memory])
    cc = np.zeros([4])
    dis2 = 1000
    cc[0] = Distance(BS_Location[0], user)
    cc[1] = Distance(BS_Location[1], user)
    cc[2] = Distance(BS_Location[2], user)
    cc[3] = Distance(BS_Location[3], user)

    aa[0] = np.where(state[0] == point)[0]
    aa[1] = np.where(state[1] == point)[0]
    aa[2] = np.where(state[2] == point)[0]
    aa[3] = np.where(state[3] == point)[0]

    for i in range(4):
        if dis2 > cc[i]:
            dis2 = cc[i]
            action = F_packet * i + np.max(aa[i])

    return int(action)

def SD_packet(Memory, BS_Location, user, state, point, F_packet, able_key):
    action = -1
    state = state.reshape([1, 4 * F_packet])
    can_action = able_key * state
    can_action = can_action.reshape([4, F_packet])

    file_in = np.where(can_action == point)

    aa = np.zeros([4, Memory])
    cc = np.zeros([4])

    cc[0] = Distance(BS_Location[0], user)
    cc[1] = Distance(BS_Location[1], user)
    cc[2] = Distance(BS_Location[2], user)
    cc[3] = Distance(BS_Location[3], user)

    dis2 = np.argmin(cc)
    kk = np.where(file_in[0] == dis2)[0]
    action = F_packet*dis2 + max(file_in[1][kk])

    return int(action)

def NO(BS_Location, user, F_packet):
    action = -1
    cc = np.zeros([4])
    dis2 = 1000
    cc[0] = Distance(BS_Location[0], user)
    cc[1] = Distance(BS_Location[1], user)
    cc[2] = Distance(BS_Location[2], user)
    cc[3] = Distance(BS_Location[3], user)

    for i in range(4):
        if dis2 > cc[i]:
            dis2 = cc[i]
            action = F_packet * i

    return int(action)