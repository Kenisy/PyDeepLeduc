''' Gives allowed bets during a game.
Bets are restricted to be from a list of predefined fractions of the pot.
'''

from Source.Settings.arguments import arguments
import torch

class BetSizing:
    def __init__(self, pot_fractions):
        ''' Constructor

        Params:
            pot_fractions: a list of fractions of the pot which are allowed 
                as bets, sorted in ascending order '''
        super().__init__()
        self.pot_fractions = pot_fractions

    def get_possible_bets(self, node):
        ''' Gives the bets which are legal at a game state.

        Params:
            node: a representation of the current game state, with fields:
                * `bets`: the number of chips currently committed by each player
                * `current_player`: the currently acting player
        Return an Nx2 tensor where N is the number of new possible game states,
        containing N sets of new commitment levels for each player '''
        current_player = node.current_player
        assert current_player == 0 or current_player == 1, 'Wrong player for bet size computation'
        opponent = 1 - node.current_player 
        opponent_bet = node.bets[opponent]

        assert node.bets[current_player] <= opponent_bet
        
        # compute min possible raise size 
        max_raise_size = arguments.stack - opponent_bet
        min_raise_size = opponent_bet - node.bets[current_player]
        min_raise_size = max(min_raise_size, arguments.ante)
        min_raise_size = min(max_raise_size, min_raise_size)
        
        if min_raise_size == 0: 
            return arguments.Tensor()
        elif min_raise_size == max_raise_size:
            out = arguments.Tensor(1, 2).fill_(opponent_bet)
            out[0][current_player] = opponent_bet + min_raise_size
            return out
        else:
            # iterate through all bets and check if they are possible
            max_possible_bets_count = self.pot_fractions.size(0) + 1 # we can always go allin 
            out = arguments.Tensor(max_possible_bets_count, 2).fill_(opponent_bet)
            
            # take pot size after opponent bet is called
            pot = opponent_bet * 2
            used_bets_count = 0
            # try all pot fractions bet and see if we can use them 
            for i in range(self.pot_fractions.size(0)): 
                raise_size = pot * self.pot_fractions[i]
                if raise_size >= min_raise_size and raise_size < max_raise_size:
                    out[used_bets_count, current_player] = opponent_bet + raise_size
                    used_bets_count = used_bets_count + 1
            # adding allin
            assert used_bets_count <= max_possible_bets_count
            out[used_bets_count, current_player] = opponent_bet + max_raise_size
            used_bets_count  = used_bets_count + 1
            return out[:used_bets_count, :]