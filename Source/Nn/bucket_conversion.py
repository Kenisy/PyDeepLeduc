''' Converts between vectors over private hands and vectors over buckets.'''

from Source.Settings.arguments import arguments
from Source.Settings.game_settings import game_settings
from Source.Nn.bucketer import Bucketer
import torch

class BucketConversion:

    def set_board(self, board):
        ''' Sets the board cards for the bucketer.

        Params:
            board: a non-empty vector of board cards'''
        self.bucketer = Bucketer()
        self.bucket_count = self.bucketer.get_bucket_count()
        self._range_matrix = arguments.Tensor(game_settings.card_count, self.bucket_count ).zero_()

        buckets = self.bucketer.compute_buckets(board)
        class_ids = torch.arange(0, self.bucket_count)
        
        if arguments.gpu: 
            buckets = buckets.cuda() 
            class_ids = class_ids.cuda()
        else:
            class_ids = class_ids.float()

        class_ids = class_ids.view(1, self.bucket_count).expand(game_settings.card_count, self.bucket_count)
        card_buckets = buckets.view(game_settings.card_count, 1).expand(game_settings.card_count, self.bucket_count)

        # finding all strength classes      
        # matrix for transformation from card ranges to strength class ranges 
        self._range_matrix[torch.eq(class_ids, card_buckets)] = 1

        # matrix for transformation form class values to card values
        self._reverse_value_matrix = self._range_matrix.T.clone()

    def card_range_to_bucket_range(self, card_range, bucket_range):
        ''' Converts a range vector over private hands to a range vector over buckets.

        @{set_board} must be called first. Used to create inputs to the neural net.

        Params:
            card_range: a probability vector over private hands
            bucket_range: a vector in which to save the resulting probability vector over buckets'''
        torch.mm(card_range, self._range_matrix, out=bucket_range)

    def bucket_value_to_card_value(self, bucket_value, card_value):
        ''' Converts a value vector over buckets to a value vector over private hands.

        @{set_board} must be called first. Used to process neural net outputs.

        Params:
            bucket_value: a vector of values over buckets
            card_value: a vector in which to save the resulting vector of values over private hands'''
        torch.mm(bucket_value, self._reverse_value_matrix, out=card_value)

    def get_possible_bucket_mask(self):
        ''' Gives a vector of possible buckets on the the board.

        @{set_board} must be called first.

        Return a mask vector over buckets where each entry is 1 if the bucket is
        valid, 0 if not'''
        mask = arguments.Tensor(1, self.bucket_count)
        card_indicator = arguments.Tensor(1, game_settings.card_count).fill_(1)
        mask = torch.mm(card_indicator, self._range_matrix)
        return mask