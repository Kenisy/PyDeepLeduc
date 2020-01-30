''' Generates neural net training data by evaluating terminal equity for poker
situations.

Evaluates terminal equity (assuming both players check/call to the end of
the game) instead of re-solving. Used for debugging.
@module data_generation_call'''
from Source.Settings.arguments import arguments
from Source.Settings.game_settings import game_settings
import torch

class M:

    def generate_data(train_data_count, valid_data_count):
        ''' Generates training and validation files by evaluating terminal
        equity for random poker situations.

        Makes two calls to @{generate_data_file}. The files are saved to 
        @{arguments.data_path}, respectively appended with `valid` and `train`.
        # 
        @param train_data_count the number of training examples to generate
        @param valid_data_count the number of validation examples to generate'''
        # valid data generation 
        file_name = arguments.data_path + 'valid'
        self:generate_data_file(valid_data_count, file_name) 
        # train data generation 
        file_name = arguments.data_path + 'train'
        self:generate_data_file(train_data_count, file_name)

    def generate_data_file(data_count, file_name):
        ''' Generates data files containing examples of random poker situations with
        associated terminal equity.

        Each poker situation is randomly generated using @{range_generator} and 
        @{random_card_generator}. For description of neural net input and target
        type, see @{net_builder}.

        @param data_count the number of examples to generate
        @param file_name the prefix of the files where the data is saved (appended
        with `.inputs`, `.targets`, and `.mask`).'''
        range_generator = RangeGenerator()
        batch_size = arguments.gen_batch_size
        assert(data_count % batch_size == 0, 'data count has to be divisible by the batch size')
        batch_count = data_count / batch_size
        bucketer = Bucketer()
        bucket_count = bucketer:get_bucket_count()
        player_count = 2
        target_size = bucket_count * player_count
        targets = arguments.Tensor(data_count, target_size)
        input_size = bucket_count * player_count + 1
        inputs = arguments.Tensor(data_count, input_size)
        mask = arguments.Tensor(data_count, bucket_count).zero_()
        bucket_conversion = BucketConversion()
        equity = TerminalEquity()
        for batch in range(batch_count):
            board = card_generator.generate_cards(game_settings.board_card_count)
            range_generator.set_board(board)
            bucket_conversion.set_board(board)
            equity.set_board(board)
            
            # generating ranges
            ranges = arguments.Tensor(player_count, batch_size, game_settings.card_count)
            for player in range(player_count):
                range_generator.generate_range(ranges[player])
            pot_sizes = arguments.Tensor(arguments.gen_batch_size, 1)
            
            # generating pot features
            pot_sizes.copy_(torch.rand(batch_size))
            
            # translating ranges to features 
            pot_feature_index =  -1
            inputs[batch * batch_size : (batch + 1) * batch_size, pot_feature_index].copy_(pot_sizes)
            for player in range(player_count):
                bucket_conversion.card_range_to_bucket_range(ranges[player], inputs[batch * batch_size : (batch + 1) * batch_size, player * bucket_count : (player + 1) * bucket_count])
            
            # computaton of values using terminal equity
            values = arguments.Tensor(player_count, batch_size, game_settings.card_count)
            for player in range(player_count):
                opponent = 1 - player
                equity.call_value(ranges[opponent], values[player])
            
            # translating values to nn targets
            for player in range(player_count):
                bucket_conversion.card_range_to_bucket_range(values[player], targets[batch * batch_size : (batch + 1) * batch_size, player * bucket_count : (player + 1) * bucket_count]])
            
            # computing a mask of possible buckets
            bucket_mask = bucket_conversion.get_possible_bucket_mask()
            mask[batch * batch_size : (batch + 1) * batch_size, :].copy_(bucket_mask.expand(batch_size, bucket_count))

        torch.save(inputs, file_name + '.inputs')
        torch.save(targets, file_name + '.targets')
        torch.save(mask, file_name + '.mask')
