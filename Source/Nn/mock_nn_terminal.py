''' Implements the same interface as @{value_nn}, but without uses terminal
equity evaluation instead of a neural net.

Can be used to replace the neural net during debugging.
@classmod mock_nn_terminal'''

from Source.Settings.arguments import arguments
from Source.Settings.game_settings import game_settings
from Source.TerminalEquity.terminal_equity import TerminalEquity
from Source.Game.card_tools import card_tools
from Source.Nn.bucketer import Bucketer
import torch

class MockNnTerminal:
    def __init__(self):
        ''' Constructor. Creates an equity matrix with entries for every possible
        pair of buckets.'''
        self.bucketer = Bucketer()
        self.bucket_count = self.bucketer.get_bucket_count()
        self.equity_matrix = arguments.Tensor(self.bucket_count, self.bucket_count).zero_()
        # filling equity matrix
        boards = card_tools.get_second_round_boards()
        self.board_count = boards.size(0)
        self.terminal_equity = TerminalEquity()
        for i in range(self.board_count): 
            board = boards[i]
            self.terminal_equity.set_board(board)
            call_matrix = self.terminal_equity.get_call_matrix()
            buckets = self.bucketer.compute_buckets(board)
            for c1 in range(game_settings.card_count): 
                for c2 in range(game_settings.card_count): 
                    b1 = buckets[c1]
                    b2 = buckets[c2]
                    if( b1 > 0 and b2 > 0 ):
                        matrix_entry = call_matrix[c1][c2]
                        self.equity_matrix[b1][b2] = matrix_entry

    def get_value(self, inputs, outputs):
        ''' Gives the expected showdown equity of the two players' ranges.
        @param inputs An NxI tensor containing N instances of neural net inputs. 
        See @{net_builder} for details of each input.
        @param outputs An NxO tensor in which to store N sets of expected showdown
        counterfactual values for each player.'''
        assert(outputs.dim() == 2 )
        bucket_count = outputs.size(1) / 2
        batch_size = outputs.size(0)
        players_count = 2
        for player in range(players_count): 
            torch.mm(inputs[:, (1 - player) * self.bucket_count : (2 - player) * self.bucket_count], self.equity_matrix, out=outputs[:, player * self.bucket_count : (player + 1) * self.bucket_count])