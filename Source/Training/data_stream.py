''' Handles the data used for neural net training and validation.'''

from Source.Settings.arguments import arguments
import torch

class Data:
    def __init__(self):
        super().__init__()

class DataStream:
    def __init__(self):
        ''' Constructor.

        Reads the data from training and validation files generated with
        @{data_generation_call.generate_data}.'''
        super().__init__()
        # loadind valid data
        self.data = Data()
        valid_prefix = arguments.data_path + 'valid'
        self.data.valid_mask= torch.load(valid_prefix + '.mask')
        self.data.valid_mask= self.data.valid_mask.repeat(1,2)
        self.data.valid_targets = torch.load(valid_prefix + '.targets')
        self.data.valid_inputs = torch.load(valid_prefix + '.inputs')
        self.valid_data_count = self.data.valid_inputs.size(0)
        assert self.valid_data_count >= arguments.train_batch_size, 'Validation data count has to be greater than a train batch size!'
        self.valid_batch_count = self.valid_data_count // arguments.train_batch_size
        # loading train data
        train_prefix = arguments.data_path + 'train'
        self.data.train_mask = torch.load(train_prefix + '.mask')
        self.data.train_mask = self.data.train_mask.repeat(1,2)
        self.data.train_inputs = torch.load(train_prefix + '.inputs')
        self.data.train_targets = torch.load(train_prefix + '.targets')
        self.train_data_count = self.data.train_inputs.size(0)
        assert self.train_data_count >= arguments.train_batch_size, 'Training data count has to be greater than a train batch size!'
        self.train_batch_count = self.train_data_count // arguments.train_batch_size
        
        # transfering data to gpu if needed
        # if arguments.gpu:
        #     for key, value in pairs(self.data) do 
        #     self.data[key] = value.cuda()

    def get_valid_batch_count(self):
        ''' Gives the number of batches of validation data.

        Batch size is defined by @{arguments.train_batch_size}.

        Return the number of batches'''
        return self.valid_batch_count

    def get_train_batch_count(self):
        ''' Gives the number of batches of training data.

        Batch size is defined by @{arguments.train_batch_size}

        Return the number of batches'''
        return self.train_batch_count

    def start_epoch(self):
        ''' Randomizes the order of training data.

        Done so that the data is encountered in a different order for each epoch.'''
        # data are shuffled each epoch 
        shuffle = torch.randperm(self.train_data_count)

        self.data.train_inputs.index_copy_(0, shuffle, self.data.train_inputs.clone())
        self.data.train_targets.index_copy_(0, shuffle, self.data.train_targets.clone())
        self.data.train_mask.index_copy_(0, shuffle, self.data.train_mask.clone())

    def get_batch(self, inputs, targets, mask, batch_index):
        ''' Returns a batch of data from a specified data set.

        Params:
            inputs: the inputs set for the given data set
            targets: the targets set for the given data set
            mask: the masks set for the given data set
            batch_index: the index of the batch to return
        Return the (inputs, targets, masks) set for the batch
        '''
        assert inputs.size(0) == targets.size(0) and inputs.size(0) == mask.size(0)
        batch_inputs = inputs[batch_index * arguments.train_batch_size : (batch_index + 1) * arguments.train_batch_size]
        batch_targets = targets[batch_index * arguments.train_batch_size : (batch_index + 1) * arguments.train_batch_size]
        batch_mask = mask[batch_index * arguments.train_batch_size : (batch_index + 1) * arguments.train_batch_size]
        return batch_inputs, batch_targets, batch_mask

    def get_train_batch(self, batch_index):
        ''' Returns a batch of data from the training set.

        Params:
            batch_index: the index of the batch to return
        Return the (inputs, targets, masks) set for the batch
        '''
        return self.get_batch(self.data.train_inputs, self.data.train_targets, self.data.train_mask, batch_index)

    def get_valid_batch(self, batch_index):
        ''' Returns a batch of data from the validation set.

        Params:
            batch_index: the index of the batch to return
        Return the (inputs, targets, masks) set for the batch
        '''
        return self.get_batch(self.data.valid_inputs, self.data.valid_targets, self.data.valid_mask, batch_index)