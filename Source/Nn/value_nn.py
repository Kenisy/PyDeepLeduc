''' Wraps the calls to the final neural net.
@classmod value_nn'''

from Source.Settings.arguments import arguments
import torch

class ValueNn:
    def __init__(self):
        super().__init__()
        net_file = arguments.model_path + arguments.value_net_name
  
        # 0.0 select the correct model cpu/gpu
        if arguments.gpu:
            net_file = net_file + '_gpu'
        else:
            net_file = net_file + '_cpu'

        # 2.0 load model  
        self.mlp = torch.load(net_file + '.pt')
        self.mlp.eval()
        print('NN architecture:')
        print(self.mlp)
        
    def get_value(inputs, output):
        ''' Gives the neural net output for a batch of inputs.
        @param inputs An NxI tensor containing N instances of neural net inputs. 
        See @{net_builder} for details of each input.
        @param output An NxO tensor in which to store N sets of neural net outputs. 
        See @{net_builder} for details of each output.'''
        output.copy_(self.mlp(inputs))
