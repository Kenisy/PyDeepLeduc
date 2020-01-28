''' Assigns hands to buckets on the given board.

For the Leduc implementation, we simply assign every possible set of
private and board cards to a unique bucket.
@classmod bucketer'''
from Source.Settings.game_settings import game_settings
from Source.Game.card_tools import card_tools
import torch
class Bucketer:

    def get_bucket_count(self):
        ''' Gives the total number of buckets across all boards.
        @return the number of buckets'''
        return game_settings.card_count * card_tools.get_boards_count()

    def compute_buckets(self, board):
        ''' Gives a vector which maps private hands to buckets on a given board.
        @param board a non-empty vector of board cards
        @return a vector which maps each private hand to a bucket index'''
        shift = card_tools.get_board_index(board) * game_settings.card_count
        # TODO recheck
        buckets = torch.arange(0, game_settings.card_count).float().add(shift)
        # impossible hands will have bucket number -1
        for i in range(board.size(0)):
            buckets[board[i].item()] = -1
        return buckets