import numpy as np

buffer_m = np.zeros([1, 80])

def Distance(x, y):  # 좌표간의 거리
    return np.sqrt(np.sum((x - y) ** 2))

def NO(BS_Location, user, F_packet):
    action = -1
    distance = np.zeros([4])
    dis2 = 1000
    distance[0] = Distance(BS_Location[0], user)
    distance[1] = Distance(BS_Location[1], user)
    distance[2] = Distance(BS_Location[2], user)
    distance[3] = Distance(BS_Location[3], user)

    for i in range(4):
        if dis2 > distance[i]:
            dis2 = distance[i]
            action = F_packet * i

    return int(action)

def one_hot_encode(file):
    temp = np.zeros(80)
    temp[file] = 1
    return temp

def Zip_funtion():  # zip distribution
    m = np.sum(np.array(range(1, 20 + 1)) ** (-0.8))
    Zip_law = (np.array(range(1, 20 + 1)) ** (-0.8)) / m
    return Zip_law

class LFU:
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

        if episode <= 300: # save a certain amount ex) 300 history
            temp_buffer = np.insert(self.buffer_m, 0, self.request_history, axis=0)
        else:
            delete_data = np.delete(self.buffer_m, -1, axis=0)
            temp_buffer = np.insert(delete_data, 0, self.request_history, axis=0)

        self.buffer_m = temp_buffer

    def CUA_LFU(self, Memory, BS_Location, user, state, point, file, F_packet):
        self.request_history[file] += 1

        hist = self.buffer_m.sum(axis=0)

        cache_state = np.zeros([4, Memory], dtype=np.int)
        distance = np.zeros([4])
        cache_hit = np.zeros([4])

        distance[0] = Distance(BS_Location[0], user)
        distance[1] = Distance(BS_Location[1], user)
        distance[2] = Distance(BS_Location[2], user)
        distance[3] = Distance(BS_Location[3], user)

        cache_state[0] = np.where(state[0] == point)[0]
        cache_state[1] = np.where(state[1] == point)[0]
        cache_state[2] = np.where(state[2] == point)[0]
        cache_state[3] = np.where(state[3] == point)[0]

        for i in range(4):
            if file in cache_state[i]:
                cache_hit[i] = 1
            else:
                cache_hit[i] = 100
        if 1 in cache_hit: # It works if you cache the request
            ee = distance * cache_hit # weight the distance
            shortest = np.argmin(ee)
            hist[cache_state[shortest]] -= 100000
            aa = np.argwhere(hist == np.amin(hist)).flatten().tolist()[-1] # LFU action
            action = F_packet * shortest + aa
        else:
            shortest = np.argmin(distance)
            hist[cache_state[shortest]] -= 100000
            aa = np.argwhere(hist == np.amin(hist)).flatten().tolist()[-1]
            action = F_packet * shortest + aa

        return int(action)

    def DUA_LFU(self,Memory, BS_Location, user, state, point, file, F_packet):
        self.request_history[file] += 1

        hist = self.buffer_m.sum(axis=0)
        cache_state = np.zeros([4, Memory], dtype=np.int)
        distance = np.zeros([4])

        distance[0] = Distance(BS_Location[0], user)
        distance[1] = Distance(BS_Location[1], user)
        distance[2] = Distance(BS_Location[2], user)
        distance[3] = Distance(BS_Location[3], user)

        cache_state[0] = np.where(state[0] == point)[0]
        cache_state[1] = np.where(state[1] == point)[0]
        cache_state[2] = np.where(state[2] == point)[0]
        cache_state[3] = np.where(state[3] == point)[0]

        shortest = np.argmin(distance)
        hist[cache_state[shortest]] -= 100000
        aa = np.argwhere(hist == np.amin(hist)).flatten().tolist()[-1] # LFU action
        action = F_packet * shortest + aa

        return int(action)