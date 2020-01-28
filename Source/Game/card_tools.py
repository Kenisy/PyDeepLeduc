''' A set of tools for basic operations on cards and sets of cards.

Several of the functions deal with "range vectors", which are probability
vectors over the set of possible private hands. For Leduc Hold'em,
each private hand consists of one card.
@module card_tools '''

from Source.Settings.game_settings import game_settings
from Source.Settings.arguments import arguments
import torch
import numpy

class M:
    def __init__(self):
        super().__init__()
        self._init_board_index_table()

    def hand_is_possible(self, hand):
        ''' Gives whether a set of cards is valid.
        @param hand a vector of cards
        @return `true` if the tensor contains valid cards and no card is repeated'''
        assert hand.min() >= 0 and hand.max() < game_settings.card_count, 'Illegal cards in hand'
        used_cards = torch.zeros(game_settings.card_count)
        for i in range(hand.size(0)): 
            used_cards[hand[i].item()] = used_cards[hand[i].item()] + 1
        return used_cards.max() < 2

    def get_possible_hand_indexes(self, board):
        ''' Gives the private hands which are valid with a given board.
        @param board a possibly empty vector of board cards
        @return a vector with an entry for every possible hand (private card), which
        is `1` if the hand shares no cards with the board and `0` otherwise'''
        out = arguments.Tensor(game_settings.card_count).fill_(0)
        if board.dim() == 0:
            out.fill_(1)
            return out

        whole_hand = arguments.IntTensor(board.size(0) + 1)
        whole_hand[:-1].copy_(board)
        for card in range(game_settings.card_count): 
            whole_hand[-1] = card
            if self.hand_is_possible(whole_hand):
                out[card] = 1
        return out

    def get_impossible_hand_indexes(self, board):
        ''' Gives the private hands which are invalid with a given board.
        @param board a possibly empty vector of board cards
        @return a vector with an entry for every possible hand (private card), which
        is `1` if the hand shares at least one card with the board and `0` otherwise'''
        out = self.get_possible_hand_indexes(board)
        out.add_(-1)
        out.mul_(-1)
        return out

    def get_uniform_range(self, board):
        ''' Gives a range vector that has uniform probability on each hand which is 
        valid with a given board.
        @param board a possibly empty vector of board cards
        @return a range vector where invalid hands have 0 probability and valid 
        hands have uniform probability'''
        out = self.get_possible_hand_indexes(board)
        out.div_(out.sum())
            
        return out

    def get_random_range(self, board, seed):
        ''' Randomly samples a range vector which is valid with a given board.
        @param board a possibly empty vector of board cards
        @param[opt] seed a seed for the random number generator
        @return a range vector where invalid hands are given 0 probability, each
        valid hand is given a probability randomly sampled from the uniform
        distribution on [0,1), and the resulting range is normalized'''
        # seed = seed or torch.random()
        out = torch.rand(game_settings.card_count)
        out.mul_(self.get_possible_hand_indexes(board))
        out.div_(out.sum())
        
        return out

    def is_valid_range(self, _range, board):
        ''' Checks if a range vector is valid with a given board.
        @param range a range vector to check
        @param board a possibly empty vector of board cards
        @return `true` if the range puts 0 probability on invalid hands and has
        total probability 1'''
        check = _range.clone()
        only_possible_hands = _range.clone().mul_(self.get_impossible_hand_indexes(board)).sum() == 0
        sums_to_one = abs(1.0 - _range.sum()) < 0.0001
        return only_possible_hands and sums_to_one

    def board_to_street(self, board):
        ''' Gives the current betting round based on a board vector.
        @param board a possibly empty vector of board cards
        @return the current betting round'''
        if board.dim() == 0 or board.size(0) == 0:
            return 1
        else:
            return 2

    def get_second_round_boards(self):
        ''' Gives all possible sets of board cards for the game.
        @return an NxK tensor, where N is the number of possible boards, and K is
        the number of cards on each board'''
        boards_count = self.get_boards_count()
        if game_settings.board_card_count == 1:
            out = arguments.Tensor(boards_count, 1)
            for card in range(game_settings.card_count): 
                out[card, 0] = card
            return out
        elif game_settings.board_card_count == 2:
            out = arguments.Tensor(boards_count, 2)
            board_idx = 0 
            for card_1 in range(game_settings.card_count): 
                for card_2 in range(card_1 + 1, game_settings.card_count): 
                    board_idx = board_idx + 1
                    out[board_idx, 0] = card_1
                    out[board_idx, 1] = card_2
            assert board_idx == boards_count, 'wrong boards count!'
            return out
        else:
            assert False, 'unsupported board size'

    def get_boards_count(self):
        ''' Gives the number of possible boards.
        @return the number of possible boards'''
        if game_settings.board_card_count == 1:
            return game_settings.card_count
        elif game_settings.board_card_count == 2: 
            return (game_settings.card_count * (game_settings.card_count - 1)) / 2
        else:
            assert False, 'unsupported board size'

    def _init_board_index_table(self):
        ''' Initializes the board index table.
        @local'''
        if game_settings.board_card_count == 1:
            self._board_index_table = torch.arange(game_settings.card_count)
        elif game_settings.board_card_count == 2:
            self._board_index_table = arguments.Tensor(game_settings.card_count, game_settings.card_count).fill_(-1)
            board_idx = 0 
            for card_1 in range(game_settings.card_count): 
                for card_2 in range(card_1 + 1, game_settings.card_count):
                    board_idx = board_idx + 1
                    self._board_index_table[card_1][card_2] = board_idx
                    self._board_index_table[card_2][card_1] = board_idx
        else:
            assert False, 'unsupported board size'

    def get_board_index(self, board):
        ''' Gives a numerical index for a set of board cards.
        @param board a non-empty vector of board cards
        @return the numerical index for the board'''
        index = self._board_index_table
        for i in range(board.size(0)):
            index = index[board[i].item()]
        assert index >= 0, index
        return index

    def normalize_range(self, board, _range):
        ''' Normalizes a range vector over hands which are valid with a given board.
        @param board a possibly empty vector of board cards
        @param range a range vector
        @return a modified version of `range` where each invalid hand is given 0
        probability and the vector is normalized'''
        mask = self.get_possible_hand_indexes(board)
        out = _range.clone().mul(mask)
        # return zero range if it all collides with board (avoid div by zero)
        if out.sum() == 0:
            return out
        out.div_(out.sum())
        return out

card_tools = M()