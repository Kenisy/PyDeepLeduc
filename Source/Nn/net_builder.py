import torch.nn as nn
import torch
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Nn.bucketer import Bucketer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        bucketer = Bucketer()
        bucket_count = bucketer.get_bucket_count()
        self.output_size = bucket_count * constants.players_count
        self.input_size = self.output_size + 1
        self.fc = nn.ModuleList([nn.Linear(self.input_size, arguments.net[0])])
        for i in range(1, len(arguments.net)):
            self.fc.append(nn.Linear(arguments.net[i-1], arguments.net[i]))
        # self.bn = nn.ModuleList([nn.BatchNorm1d(layer) for layer in arguments.net])
        self.fc1 = nn.Linear(arguments.net[-1], self.output_size)
    
    def forward(self, x):
        feedforward = x.clone()
        ranges = x.narrow(1, 0, self.output_size)

        for i in range(len(self.fc)):
            feedforward = self.fc[i](feedforward)
            # feedforward = self.bn[i](feedforward)
            feedforward = nn.ReLU()(feedforward)

        feedforward = self.fc1(feedforward)
        estimated_value = torch.bmm(feedforward.unsqueeze(1), ranges.unsqueeze(2)).squeeze(2)
        estimated_value = estimated_value.repeat(1, self.output_size)
        estimated_value = torch.mul(estimated_value, -0.5)
        final_mlp = torch.add(feedforward, estimated_value)
        return final_mlp
        