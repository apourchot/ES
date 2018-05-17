import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FFIndiv(nn.Module):

    def __init__(self, obs_space, action_space, hidden_size=32):

        super(FFIndiv, self).__init__()

        self.input_size = int(np.prod(obs_space[1]))
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        if action_space[0] == "Discrete":
            self.discrete = True
            self.output_size = action_space[1]
        else:
            self.discrete = False
            self.output_size = action_space[1][0]
        self.fc3 = nn.Linear(hidden_size, self.output_size)

        xavier_init = torch.nn.init.xavier_uniform_
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        state_dict = self.state_dict()
        cpt = 0

        # putting parameters in the right shape
        for k, v in zip(state_dict.keys(), state_dict.values()):
            tmp = np.product(v.size())
            state_dict[k] = torch.from_numpy(params[cpt:cpt+tmp]).view(v.size())
            cpt += tmp

        # setting parameters of the network
        self.load_state_dict(state_dict)

    def get_params(self):
        """
        Params that should be trained on.
        """
        return np.hstack([v.numpy().flatten() for v in
                            self.state_dict().values()])


    def get_action(self, obs):
        """
        Get the action to take given the current
        observation
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).type(torch.FloatTensor)
            tmp = self.forward(obs)
            if self.discrete:
                return torch.argmax(F.softmax(tmp, dim=1)).numpy()
            else:
                return tmp.numpy()


    def forward(self, x):
        """
        Forward pass
        """
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
