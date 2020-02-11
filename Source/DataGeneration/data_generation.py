''' Generates neural net training data by solving random poker situations.'''
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Tree.tree_builder import TreeNode
from Source.DataGeneration.random_card_generator import card_generator
from Source.DataGeneration.range_generator import RangeGenerator
from Source.Nn.bucketer import Bucketer
from Source.Nn.bucket_conversion import BucketConversion
from Source.Lookahead.resolving import Resolving
from tqdm import tqdm
import os.path
import torch
import time

class M:

    def generate_data(self, train_data_count, valid_data_count):
        ''' Generates training and validation files by sampling random poker
        situations and solving them.

        Makes two calls to @{generate_data_file}. The files are saved to 
        @{arguments.data_path}, respectively appended with `valid` and `train`.
        
        Params:
            train_data_count: the number of training examples to generate
            valid_data_count: the number of validation examples to generate'''
        # valid data generation 
        file_name = arguments.data_path + 'valid'
        timer = time.time()
        print('Generating validation data ...')
        self.generate_data_file(valid_data_count, file_name)
        print(f'valid gen time: {time.time() - timer}')
        timer = time.time()
        # train data generation 
        print('Generating training data ...')
        file_name = arguments.data_path + 'train'
        self.generate_data_file(train_data_count, file_name) 
        print(f'Generation time: {time.time() - timer}')
        print('Done')

    def generate_data_file(self, data_count, file_name):
        ''' Generates data files containing examples of random poker situations with
        counterfactual values from an associated solution.

        Each poker situation is randomly generated using @{range_generator} and 
        @{random_card_generator}. For description of neural net input and target
        type, see @{net_builder}.

        Params:
            data_count: the number of examples to generate
            file_name: the prefix of the files where the data is saved (appended 
                with `.inputs`, `.targets`, and `.mask`).'''
        range_generator = RangeGenerator()
        batch_size = arguments.gen_batch_size
        assert data_count % batch_size == 0, 'data count has to be divisible by the batch size'
        batch_count = data_count // batch_size
        bucketer = Bucketer()
        bucket_count = bucketer.get_bucket_count()
        target_size = bucket_count * constants.players_count
        targets = arguments.Tensor(data_count, target_size)
        input_size = bucket_count * constants.players_count + 1
        inputs = arguments.Tensor(data_count, input_size)
        mask = arguments.Tensor(data_count, bucket_count).zero_()
        bucket_conversion = BucketConversion()
        for batch in tqdm(range(batch_count)):
            board = card_generator.generate_cards(game_settings.board_card_count)
            range_generator.set_board(board)
            bucket_conversion.set_board(board)
            
            # generating ranges
            ranges = arguments.Tensor(constants.players_count, batch_size, game_settings.card_count)
            for player in range(constants.players_count):
                range_generator.generate_range(ranges[player])
            
            # generating pot sizes between ante and stack - 0.1
            min_pot = arguments.ante
            max_pot = arguments.stack - 0.1
            pot_range = max_pot - min_pot
            
            random_pot_sizes = torch.rand(arguments.gen_batch_size, 1).mul(pot_range).add(min_pot)
            
            # pot features are pot sizes normalized between (ante/stack,1)
            pot_size_features = random_pot_sizes.clone().mul(1/arguments.stack)
            
            # translating ranges to features 
            pot_feature_index =  -1
            inputs[batch * batch_size : (batch + 1) * batch_size, pot_feature_index].copy_(pot_size_features.squeeze(1))
            for player in range(constants.players_count):
                bucket_conversion.card_range_to_bucket_range(ranges[player], inputs[batch * batch_size : (batch + 1) * batch_size, player * bucket_count : (player + 1) * bucket_count])
            
            # computaton of values using re-solving
            values = arguments.Tensor(constants.players_count, batch_size, game_settings.card_count)
            for i in range(batch_size): 
                resolving = Resolving()
                current_node = TreeNode()

                current_node.board = board
                current_node.street = 2
                current_node.current_player = constants.players.P1
                pot_size = pot_size_features[i][0] * arguments.stack
                current_node.bets = arguments.Tensor([pot_size, pot_size])
                p1_range = ranges[0][i]
                p2_range = ranges[1][i]
                resolving.resolve_first_node(current_node, p1_range, p2_range)
                root_values = resolving.get_root_cfv_both_players()
                root_values.mul_(1/pot_size)
                values[:, i, :].copy_(root_values)
            
            # translating values to nn targets
            for player in range(constants.players_count):
                bucket_conversion.card_range_to_bucket_range(values[player], targets[batch * batch_size : (batch + 1) * batch_size, player * bucket_count : (player + 1) * bucket_count])
            # computing a mask of possible buckets
            bucket_mask = bucket_conversion.get_possible_bucket_mask()
            mask[batch * batch_size : (batch + 1) * batch_size, :].copy_(bucket_mask.expand(batch_size, bucket_count))
        
        if os.path.exists(file_name + '.inputs'):
            saved_inputs = torch.load(file_name + '.inputs')
            saved_targets = torch.load(file_name + '.targets')
            saved_mask = torch.load(file_name + '.mask')
            inputs = torch.cat((saved_inputs, inputs), 0)
            targets = torch.cat((saved_targets, targets), 0)
            mask = torch.cat((saved_mask, mask), 0)
        torch.save(inputs, file_name + '.inputs')
        torch.save(targets, file_name + '.targets')
        torch.save(mask, file_name + '.mask')

data_generation = M()