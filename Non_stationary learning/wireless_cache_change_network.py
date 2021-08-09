import numpy as np

class cache_replacement:

    def __init__(self, pop):
        self.Num_file = 20 # Num_file
        self.Num_packet = 4 # Num_packet
        self.Memory = 16 # Size_memory
        self.Small_cell = 4 # Num_SBSs
        self.x_max = 250
        self.y_max = 250
        self.F_packet = self.Num_file * self.Num_packet # Total number of chunks
        self.alpha = 0.8 # Popularity
        self.BH_path = 3.6 # Backhaul pathloss
        self.AC_path = 3.6 # Access pathloss
        self.Transmission_Power = 10**9
        self.user_location = np.zeros([2]) # user location
        self.BS = np.zeros([self.Small_cell, self.Memory], dtype=int) # SBS Memory state
        self.BS_Location = np.array([[(-1 * self.x_max / 2.0), (self.y_max / 2.0)],
                                    [(self.x_max / 2.0), (self.y_max / 2.0)],
                                    [(-1 * self.x_max / 2.0), (-1 * self.y_max / 2.0)],
                                    [(self.x_max / 2.0), (-1 * self.y_max / 2.0)]]) # SBS location
        self.state = np.zeros([self.Small_cell, self.F_packet]) # input_data
        self.Transmit_Rate = 1.0
        self.M_S_error = 0.1 # Macro->Small init error
        self.Macro_BS = 5 # Macro-> Small latency
        self.Small_BS = 1 # Small-> User latency
        self.cost, self.fail, self.count = 0, 0, 0
        self.point = 1 # state point
        self.Zip_law = [] # zip probability
        self.Setting = tuple(range(0, self.F_packet, self.Num_packet)) # popularity set
        self.file_request = []
        self.error = 0
        self.TTL_time = 100
        self.hit = 0
        self.change = pop
        self.change_setting = np.array(range(0, self.F_packet, self.Num_packet))

    def change_pop(self): # non-stationary func
        FI = self.change_setting[:self.change]
        aaa = np.delete(self.change_setting, range(self.change))
        self.change_setting = np.append(aaa, FI)
        self.Setting = tuple(self.change_setting)
        print(self.Setting)

    def Zip_funtion(self): # generate zip distribution
        m = np.sum(np.array(range(1, self.Num_file+1))**(-self.alpha))
        self.Zip_law = (np.array(range(1, self.Num_file+1))**(-self.alpha)) / m

    def reset(self): # init episode
        self.BS = np.zeros([self.Small_cell, self.Memory], dtype=int)
        self.BS[0] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.BS[1] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.BS[2] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.BS[3] = np.random.choice(self.F_packet, self.Memory, replace=False)
        self.state = np.zeros([self.Small_cell, self.F_packet])
        for i in range(self.Small_cell):
            self.state[i][self.BS[i]] = self.point

        d = self.Distance(self.BS_Location[0], [0, 0])
        self.MS_error = self.Probabilistic_BH(d)

        self.user_location = np.random.uniform(-1*self.x_max, self.x_max, (1, 2))[0]
        self.cost, self.fail, self.count, self.hit = 0, 0, 0, 0
        self.file_request = np.random.choice(self.Setting, 1001, p=self.Zip_law)

        state = self.flat(self.user_location, self.file_request[0])
        return state

    def cache_revalidation(self, ap_num, file):
        if self.state[ap_num][file] == self.point:
            return 0
        else:
            return 1

    def cache_miss(self, ap_num, file_del, file):
        self.state[ap_num][file_del] = 0
        self.state[ap_num][file] = self.point

    def TTL(self, n):
        if n * self.Small_BS >= self.TTL_time:
            self.fail += 1
            return self.TTL_time
        else:
            return n * self.Small_BS

    def Distance(self, x, y): # Distance
        return np.sqrt(np.sum((x - y)**2))

    def Probabilistic_BH(self, d):
        prob = 1.0 - np.exp(-1 * ((2**self.Transmit_Rate - 1)*d**self.BH_path) / self.Transmission_Power)
        return prob

    def Probabilistic_AC(self, d):
        prob = 1.0 - np.exp(-1 * ((2**self.Transmit_Rate - 1)*d**self.AC_path) / self.Transmission_Power)
        return prob

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
            m = np.random.geometric(p=(1 - self.M_S_error))
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
        return np.argmax(table[0])

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

    def memory_state(self):
        state = np.array([np.where(self.state[0] == self.point)[0], np.where(self.state[1] == self.point)[0],
                          np.where(self.state[2] == self.point)[0], np.where(self.state[3] == self.point)[0]])
        return state


