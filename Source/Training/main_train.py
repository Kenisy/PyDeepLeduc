''' Script that trains the neural network.

Uses data previously generated with @{data_generation_call}.
@script main_train'''
import sys
sys.path.append(sys.path[0] + '/../../')
from Source.Training.data_stream import DataStream
from Source.Training.train import train
from Source.Nn.net_builder import Net
from Source.Settings.arguments import arguments

if __name__ == "__main__":
    network = Net()
    data_stream = DataStream()
    train.train(network, data_stream, arguments.epoch_count)