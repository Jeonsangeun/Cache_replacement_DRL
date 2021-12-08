import numpy as np
import collections

buffer_m = np.zeros([1, 80])

def Distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def CD(Memory, BS_Location, user, state, point, file, F_packet):
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
        action = F_packet * dis2 + max(file_in[1][kk])

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
    action = F_packet * dis2 + max(file_in[1][kk])

    return int(action)


def NO(Memory, BS_Location, user, state, point, F_packet):
    action = -1
    aa = np.zeros([4, Memory])
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

def one_hot_encode(file):
    temp = np.zeros(80)
    temp[file] = 1
    return temp

def Zip_funtion():
    m = np.sum(np.array(range(1, 20 + 1)) ** (-0.8))
    Zip_law = (np.array(range(1, 20 + 1)) ** (-0.8)) / m
    return Zip_law

def search():
    hist, bin = np.histogram(buffer_m, np.arange(81))
    print(hist)

class pop_change:
    def __init__(self):
        self.buffer_m = np.zeros([1, 80])
        self.request_history = np.zeros(80)

    def init_change(self):
        Setting = tuple(range(0, 80, 4))
        file_request = np.random.choice(Setting, 1000, p=Zip_funtion())
        first_buffer, bin = np.histogram(file_request, range(0, 81))
        temp = np.zeros(80)
        for i in range(20):
            for k in range(4):
                temp[4 * i + k] = first_buffer[4 * i]
        self.buffer_m[0] = temp

    def init_episode(self):
        self.request_history = np.zeros(80)

    def buffer_his(self, episode):

        if episode < 299:
            aa = np.insert(self.buffer_m, 0, self.request_history, axis=0)
        else:
            bb = np.delete(self.buffer_m, -1, axis=0)
            aa = np.insert(bb, 0, self.request_history, axis=0)

        self.buffer_m = aa

    def CD_pop(self, Memory, BS_Location, user, state, point, file, F_packet):
        self.request_history[file] += 1

        hist = self.buffer_m.sum(axis=0)

        aa = np.zeros([4, Memory], dtype=np.int)
        cc = np.zeros([4])
        dd = np.zeros([4])

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
                dd[i] = 1
            else:
                dd[i] = 100
        if 1 in dd:
            ee = cc * dd
            shortest = np.argmin(ee)
            hist[aa[shortest]] -= 100000
            aa = np.argwhere(hist == np.amin(hist)).flatten().tolist()[-1]
            action = F_packet * shortest + aa
        else:
            shortest = np.argmin(cc)
            hist[aa[shortest]] -= 100000
            aa = np.argwhere(hist == np.amin(hist)).flatten().tolist()[-1]
            action = F_packet * shortest + aa

        return int(action)

    def SD_pop(self,Memory, BS_Location, user, state, point, file, F_packet):
        self.request_history[file] += 1

        hist = self.buffer_m.sum(axis=0)
        aa = np.zeros([4, Memory], dtype=np.int)
        cc = np.zeros([4])

        cc[0] = Distance(BS_Location[0], user)
        cc[1] = Distance(BS_Location[1], user)
        cc[2] = Distance(BS_Location[2], user)
        cc[3] = Distance(BS_Location[3], user)

        aa[0] = np.where(state[0] == point)[0]
        aa[1] = np.where(state[1] == point)[0]
        aa[2] = np.where(state[2] == point)[0]
        aa[3] = np.where(state[3] == point)[0]

        shortest = np.argmin(cc)
        hist[aa[shortest]] -= 100000
        aa = np.argwhere(hist == np.amin(hist)).flatten().tolist()[-1]
        action = F_packet * shortest + aa

        return int(action)




