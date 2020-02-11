''' Evaluates hand strength in Leduc Hold'em and variants.

Works with hands which contain two or three cards, but assumes that
the deck contains no more than two cards of each rank (so three-of-a-kind
is not a possible hand).

Hand strength is given as a numerical value, where a lower strength means
a stronger hand: high pair < low pair < high card < low card
'''

from Source.Settings.game_settings import game_settings
from Source.Game.card_tools import card_tools
from Source.Game.card_to_string_conversion import card_to_string
from Source.Settings.arguments import arguments
import torch

class M:
    def evaluate_two_card_hand(self, hand_ranks):
        ''' Gives a strength representation for a hand containing two cards.

        Params:
            hand_ranks: the rank of each card in the hand
        Return the strength value of the hand'''
        # check for the pair 
        hand_value = None
        if hand_ranks[0] == hand_ranks[1]:
            # hand is a pair
            hand_value = hand_ranks[0]
        else:
            # hand is a high card    
            hand_value = (hand_ranks[0] + 1) * game_settings.rank_count + hand_ranks[1]
        return hand_value

    def evaluate_three_card_hand(self, hand_ranks):
        ''' Gives a strength representation for a hand containing three cards.

        Params:
            hand_ranks: the rank of each card in the hand
        Return the strength value of the hand'''
        hand_value = None
        # check for the pair 
        if hand_ranks[0] == hand_ranks[1]: 
            # paired hand, value of the pair goes first, value of the kicker goes second
            hand_value = hand_ranks[0] * game_settings.rank_count + hand_ranks[2]
        elif hand_ranks[1] == hand_ranks[2]: 
            #paired hand, value of the pair goes first, value of the kicker goes second
            hand_value = hand_ranks[1] * game_settings.rank_count + hand_ranks[0]
        else:
            # hand is a high card    
            hand_value = hand_ranks[0] * game_settings.rank_count * game_settings.rank_count + hand_ranks[1] * game_settings.rank_count + hand_ranks[2]
        return hand_value

    def evaluate(self, hand, impossible_hand_value=-1):
        ''' Gives a strength representation for a two or three card hand.

        Params:
            hand: a vector of two or three cards
            impossible_hand_value [opt]: the value to return if the hand is invalid
        Return the strength value of the hand, or `impossible_hand_value` if the 
        hand is invalid'''
        assert hand.max() < game_settings.card_count and hand.min() >= 0, 'hand does not correspond to any cards'
        impossible_hand_value = impossible_hand_value
        if not card_tools.hand_is_possible(hand):
            return impossible_hand_value
        # we are not interested in the hand suit - we will use ranks instead of cards
        hand_ranks = hand.clone()
        for i in range(hand_ranks.size(0)): 
            hand_ranks[i] = card_to_string.card_to_rank(hand_ranks[i])
        hand_ranks, _ = hand_ranks.sort()
        if hand.size(0) == 2:
            return self.evaluate_two_card_hand(hand_ranks)
        elif hand.size(0) == 3:
            return self.evaluate_three_card_hand(hand_ranks)
        else:
            assert(False, 'unsupported size of hand!' )

    def batch_eval(self, board, impossible_hand_value=-1):
        ''' Gives strength representations for all private hands on the given board.

        Params:
            board: a possibly empty vector of board cards
            impossible_hand_value: the value to assign to hands which are invalid on the board
        Return a vector containing a strength value or `impossible_hand_value` for
        every private hand'''
        hand_values = arguments.Tensor(game_settings.card_count).fill_(-1)
        if board.dim() == 0:
            for hand in range(game_settings.card_count): 
                hand_values[hand] = (hand // game_settings.suit_count) + 1
        else:
            board_size = board.size(0)
            assert board_size == 1 or board_size == 2, 'Incorrect board size for Leduc'
            whole_hand = arguments.IntTensor(board_size + 1)
            whole_hand[:-1].copy_(board)
            for card in range(game_settings.card_count): 
                whole_hand[-1] = card
                hand_values[card] = self.evaluate(whole_hand, impossible_hand_value)
        return hand_values

evaluator = M()