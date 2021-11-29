import numpy as np

class cache_replacement:

    def __init__(self, coverage, Zipf_ex, Mem):
        self.Num_file = 20 #파일 수
        self.Num_packet = 4 #파일당 패킷 수
        self.Memory = Mem #메모리 크기
        self.Small_cell = 4
        self.x_max = coverage #이용자 최대 위치(횡축)
        self.y_max = coverage #이용자 최대 위치(종축)
        self.F_packet = self.Num_file * self.Num_packet #총 패킷 수
        self.alpha = Zipf_ex #zip 알파
        self.BH_path = 3.6
        self.AC_path = 3.6
        self.Transmission_Power = 10**9 #송신전력
        self.user_location = np.zeros([2])  #유저의 위치
        self.BS = np.zeros([self.Small_cell, self.Memory], dtype=int) #BS 송신소 상태
        self.BS_Location = np.array([[(-1 * self.x_max / 2.0), (self.y_max / 2.0)],
                                    [(self.x_max / 2.0), (self.y_max / 2.0)],
                                    [(-1 * self.x_max / 2.0), (-1 * self.y_max / 2.0)],
                                    [(self.x_max / 2.0), (-1 * self.y_max / 2.0)]]) #BS0의 위치
        self.state = np.zeros([self.Small_cell, self.F_packet]) #state 설정
        self.Transmit_Rate = 1.0 #전송률
        self.M_S_error = 0.1 #Macro->Small 에러율
        self.M_S_distance = np.sqrt(np.sum((self.BS_Location[0] - [0, 0]) ** 2))
        self.Macro_BS = 5 #Macro->Small cost
        self.Small_BS = 1 #Small->Macro cost
        self.cost, self.fail, self.count = 0, 0, 0 #1episode 당 cost #패킷 리퀘스트 카운트
        self.point = 1 #state 설정값
        self.Zip_law = [] #zip 분포
        self.Setting = tuple(range(0, self.F_packet, self.Num_packet)) #zip 분포 파일
        self.file_request = []
        self.error = 0
        self.TTL_time = 100
        self.hit = 0

    def Zip_funtion(self): #zip 분포 생성
        m = np.sum(np.array(range(1, self.Num_file+1))**(-self.alpha))
        self.Zip_law = (np.array(range(1, self.Num_file+1))**(-self.alpha)) / m

    def reset(self): #reset
        self.BS = np.zeros([self.Small_cell, self.Memory], dtype=int)
        self.BS[0] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.BS[1] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.BS[2] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.BS[3] = np.random.choice(self.F_packet, self.Memory, replace=False)
        # self.BS[0] = np.arange(self.Memory)
        # self.BS[1] = np.arange(self.Memory)
        # self.BS[2] = np.arange(self.Memory)
        # self.BS[3] = np.arange(self.Memory)

        # self.BS = np.load("BS.npy")
        self.state = np.zeros([self.Small_cell, self.F_packet])
        for i in range(self.Small_cell):
            self.state[i][self.BS[i]] = self.point

        self.MS_error = self.Probabilistic_BH(self.M_S_distance)

        self.user_location = np.random.uniform(-1*self.x_max, self.x_max, (1, 2))[0]
        self.cost, self.fail, self.count, self.hit = 0, 0, 0, 0
        self.file_request = np.random.choice(self.Setting, 10001, p=self.Zip_law) #패킷 추출
        state = self.flat(self.user_location, self.file_request[0]) #, np.array([0, 0])
        return state

    def cache_revalidation(self, ap_num, file): # 요청 컨텐츠가 memory에 있는지?
        if self.state[ap_num][file] == self.point:
            return 0
        else:
            return 1

    def cache_miss(self, ap_num, file_del, file): # 컨텐츠 교체
        self.state[ap_num][file_del] = 0
        self.state[ap_num][file] = self.point

    def TTL(self, n):
        if n * self.Small_BS >= self.TTL_time:
            self.fail += 1
            return self.TTL_time
        else:
            return n * self.Small_BS

    def Distance(self, x, y): #좌표간의 거리
        return np.sqrt(np.sum((x - y)**2))

    def Probabilistic_BH(self, d):
        prob = 1.0 - np.exp(-1 * ((2**self.Transmit_Rate - 1)*d**self.BH_path) / self.Transmission_Power)
        return prob

    def Probabilistic_AC(self, d):
        prob = 1.0 - np.exp(-1 * ((2**self.Transmit_Rate - 1)*d**self.AC_path) / self.Transmission_Power)
        return prob

    #action 선택
    def random_action(self):
        memory = np.reshape(self.state, [1, 4*self.F_packet])
        aa = np.where(memory[0] == self.point)[0]
        n = np.random.randint(len(aa))
        return aa[n]

    def step(self, action, file, user):
        cost = 0
        reward = 0.0
        done = False
        ap_index = int(action // self.F_packet)
        file_refresh = int(action % self.F_packet)

        d = self.Distance(self.BS_Location[ap_index], user)
        if self.cache_revalidation(ap_index, file) == 0:
            n = np.random.geometric(p=(1 - self.Probabilistic_AC(d)))
            time = self.TTL(n)
            cost += time
            reward -= time
            self.hit += 1
            self.count += 1
        else:
            self.cache_miss(ap_index, file_refresh, file)
            m = np.random.geometric(p=(1 - self.MS_error))
            cost += m * self.Macro_BS
            reward -= m * self.Macro_BS

            n = np.random.geometric(p=(1 - self.Probabilistic_AC(d)))
            time = self.TTL(n)
            cost += time
            reward -= time

            self.count += 1

        if self.count % self.Num_packet == 0:
            file = self.file_request[self.count // self.Num_packet]
            user = np.random.uniform(-1 * self.x_max, self.x_max, (1, 2))[0]
        else:
            file += 1

        new_state = self.flat(user, file)

        self.cost += cost
        if self.count == 1000 * self.Num_packet:
            done = True

        return new_state, reward, done, file, user

    def action_select(self, Q, Noise):
        able = np.reshape(self.state, [1, self.Small_cell * self.F_packet])
        table = Q * able
        table = np.where(table == 0, -1000, table + Noise)
        # print("임시:", table)
        return np.argmax(table[0])

    # def flat(self, user, file):
    #     prop = np.zeros([1, self.F_packet])
    #     d = np.zeros([4])
    #     z = np.zeros([self.F_packet])
    #     z[file] = self.point
    #     for i in range(4):
    #         temp = np.zeros([1, self.F_packet])
    #         d[i] = np.array(self.Distance(self.BS_Location[i], user))
    #         temp[0] = (1 - self.Probabilistic(d[i])/100)
    #         prop = np.vstack([prop, temp])
    #     result = np.reshape(self.state, [4, self.F_packet])
    #     result = np.vstack([result, z])
    #     result = np.vstack([result, prop[1:]])
    #     return result

    def flat(self, user, file):
        d = np.zeros([self.Small_cell])
        z = np.zeros([self.F_packet])
        p = np.zeros([self.F_packet])
        z[file] = self.point
        temp = self.state * 1
        result = np.reshape(temp, [self.Small_cell, self.F_packet])
        for i in range(self.Small_cell):
            d[i] = np.array(self.Distance(self.BS_Location[i], user))
            p[i] = (1 - self.Probabilistic_AC(d[i]))
        result = np.vstack([result, z])
        result = np.vstack([result, p])
        return result

    # def flat(self, user, file):
    #     d = np.zeros([4])
    #     z = np.zeros([self.F_packet])
    #     z[file] = self.point
    #     temp = self.state*1
    #     result = np.reshape(temp, [4, self.F_packet])
    #     for i in range(4):
    #         d[i] = np.array(self.Distance(self.BS_Location[i], user))
    #         p = (1 - self.Probabilistic(d[i])/100)
    #         result[i] = result[i] * p
    #     result = np.vstack([result, z])
    #     return result

    def print(self):
        print(np.where(self.state[0] == self.point)[0])
        print(np.where(self.state[1] == self.point)[0])
        print(np.where(self.state[2] == self.point)[0])
        print(np.where(self.state[3] == self.point)[0])

    def memory_state(self):
        state = np.array([np.where(self.state[0] == self.point)[0], np.where(self.state[1] == self.point)[0],
                          np.where(self.state[2] == self.point)[0], np.where(self.state[3] == self.point)[0]])
        return state


