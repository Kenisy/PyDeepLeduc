''' Script that generates training and validation files.'''
import sys
sys.path.append(sys.path[0] + '/../../')
from Source.Settings.arguments import arguments
from Source.DataGeneration.data_generation import data_generation

if __name__ == "__main__": 
    data_generation.generate_data(arguments.train_data_count, arguments.valid_data_count)