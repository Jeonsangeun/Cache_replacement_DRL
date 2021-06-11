import random as rd
import torch
import torch.nn as nn
import collections

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else "cpu")
buffer_limit = 1000000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = rd.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

# DQN-FCN DNN model
class Qnet_FCN(nn.Module):
    def __init__(self, input_size, node, output_size):
        super(Qnet_FCN, self).__init__()
        self.input_size = input_size
        self.node = node
        self.output_size = output_size

        fc1 = nn.Linear(self.input_size, self.node)
        fc2 = nn.Linear(self.node, self.output_size)
        fc3 = nn.Linear(self.output_size, self.output_size)

        nn.init.xavier_uniform_(fc1.weight.data)
        fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(fc2.weight.data)
        fc2.bias.data.fill_(0)
        nn.init.xavier_uniform_(fc3.weight.data)
        fc3.bias.data.fill_(0)

        self.fc = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3).to(device)

    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        x_s = x[:, :5] * 1
        res1 = x_s.reshape(-1, 400)

        feature = torch.cat([prop, res1], dim=1)

        out = self.fc(feature)
        return out

# create a Predict function to separate CPU and GPU operations for every class
def Predict_Qnet_FCN(model, x):
    prop = x[:, 5, :4] * 1
    x_s = x[:, :5] * 1
    res1 = x_s.reshape(-1, 400)
    feature = torch.cat([prop, res1], dim=1)

    out = model.fc(feature)
    return out

# Proposed scheme DNN model
class Qnet_v6(nn.Module):
    def __init__(self, Num_packet, Num_file, F_packet, node, output_size):
        super(Qnet_v6, self).__init__()
        self.node = node
        self.Num_packet = Num_packet
        self.Num_file = Num_file
        self.F_packet = F_packet
        self.output_size = output_size
        conv1 = nn.Conv2d(1, 20, kernel_size=(1, self.Num_packet),
                          stride=(1, self.Num_packet))
        # pool1 = nn.MaxPool1d(1)

        conv2 = nn.Conv2d(20, 20, kernel_size=(4, 1), stride=(4, 1))
        # pool2 = nn.MaxPool1d(1)

        self.conv_module = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU()).to(device)
        self.relu = nn.LeakyReLU(inplace=True).to(device)

        self.Lc_error = nn.Linear(1, 1, bias=False).to(device)
        self.Lc_request = nn.Linear(1, 1, bias=False).to(device)

        fc1 = nn.Linear(self.node, self.node)
        fc2 = nn.Linear(self.node, self.output_size)
        fc3 = nn.Linear(self.output_size, self.output_size)
        self.fully_module1 = nn.Sequential(fc1, self.relu, fc2, self.relu, fc3).to(device)


        nn.init.xavier_uniform_(fc1.weight.data)
        fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(fc2.weight.data)
        fc2.bias.data.fill_(0)
        nn.init.xavier_uniform_(fc3.weight.data)
        fc3.bias.data.fill_(0)


    def forward(self, x):
        x = x.to(device)
        prop = x[:, 5, :4] * 1
        req = x[:, 4] * 1
        x_s = x[:, :4] * 1
        x_s[:, 0] += self.Lc_error(prop[:, 0].unsqueeze(1))
        x_s[:, 1] += self.Lc_error(prop[:, 1].unsqueeze(1))
        x_s[:, 2] += self.Lc_error(prop[:, 2].unsqueeze(1))
        x_s[:, 3] += self.Lc_error(prop[:, 3].unsqueeze(1))

        x_s += torch.mul(self.Lc_request(torch.ones(req.shape[0], 1).to(device)), req).unsqueeze(1)

        sbs_s = x_s.reshape([-1, 1, 4, self.F_packet])

        p_s = self.conv_module(sbs_s)

        out = p_s.reshape(-1, self.node)
        out = self.fully_module1(out)

        return out


def Predict_Qnet_v6(model, x):
    prop = x[:, 5, :4] * 1
    req = x[:, 4] * 1
    x_s = x[:, :4] * 1
    x_s[:, 0] += model.Lc_error(prop[:, 0].unsqueeze(1))
    x_s[:, 1] += model.Lc_error(prop[:, 1].unsqueeze(1))
    x_s[:, 2] += model.Lc_error(prop[:, 2].unsqueeze(1))
    x_s[:, 3] += model.Lc_error(prop[:, 3].unsqueeze(1))

    x_s += (model.Lc_request(torch.ones(1)) * req)
    sbs_s = x_s.reshape([-1, 1, 4, model.F_packet])

    p_s = model.conv_module1(sbs_s)

    out = p_s.reshape(-1, 400)
    out = model.fully_module(out)

    return out