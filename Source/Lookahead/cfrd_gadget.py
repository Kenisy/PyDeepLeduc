''' Uses the the CFR-D gadget to generate opponent ranges for re-solving.

See [Solving Imperfect Information Games Using Decomposition](http.//poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf)
'''
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Game.card_tools import card_tools
from Source.Settings.game_settings import game_settings
import torch

class CFRDGadget:
    def __init__(self, board, player_range, opponent_cfvs):
        ''' Constructor

        Params:
            board: board card
            player_range: an initial range vector for the opponent
            opponent_cfvs: the opponent counterfactual values vector used for re-solving'''
        super().__init__()
        assert(board != None)

        self.input_opponent_range = player_range.clone()
        self.input_opponent_value = opponent_cfvs.clone()

        self.curent_opponent_values = arguments.Tensor(game_settings.card_count)
        
        self.regret_epsilon = 1.0/100000000
        
        # 2 stands for 2 actions: play/terminate
        self.opponent_reconstruction_regret = arguments.Tensor(2, game_settings.card_count)

        self.play_current_strategy = arguments.Tensor(game_settings.card_count).fill_(0)
        self.terminate_current_strategy = arguments.Tensor(game_settings.card_count).fill_(1)

        # holds achieved CFVs at each iteration so that we can compute regret
        self.total_values = arguments.Tensor(game_settings.card_count)

        self.terminate_regrets = arguments.Tensor(game_settings.card_count).fill_(0)
        self.play_regrets = arguments.Tensor(game_settings.card_count).fill_(0)

        # init range mask for masking out impossible hands
        self.range_mask = card_tools.get_possible_hand_indexes(board)

        self.total_values_p2 = None
        self.play_current_regret = None
        self.terminate_current_regret = None

    def compute_opponent_range(self, current_opponent_cfvs, iteration):
        ''' Uses one iteration of the gadget game to generate an opponent range for
        the current re-solving iteration.

        Params:
            current_opponent_cfvs: the vector of cfvs that the opponent receives 
                with the current strategy in the re-solve game
            iteration: the current iteration number of re-solving
        Return the opponent range vector for this iteration'''
        play_values = current_opponent_cfvs
        terminate_values = self.input_opponent_value

        # 1.0 compute current regrets  
        self.total_values = torch.mul(play_values, self.play_current_strategy)
        self.total_values_p2 = torch.mul(terminate_values, self.terminate_current_strategy)
        self.total_values.add_(self.total_values_p2)
        
        self.play_current_regret = play_values.clone()
        self.play_current_regret.sub_(self.total_values.view(self.play_current_regret.shape))
        
        self.terminate_current_regret = terminate_values.clone()
        self.terminate_current_regret.sub_(self.total_values.view(self.terminate_current_regret.shape))

        # 1.1 cumulate regrets
        self.play_regrets.add_(self.play_current_regret.view(self.play_regrets.shape))
        self.terminate_regrets.add_(self.terminate_current_regret)
        
        # 2.0 we use cfr+ in reconstruction  
        self.terminate_regrets.clamp_(self.regret_epsilon, constants.max_number)
        self.play_regrets.clamp_(self.regret_epsilon, constants.max_number)

        self.play_possitive_regrets = self.play_regrets
        self.terminate_possitive_regrets = self.terminate_regrets

        # 3.0 regret matching
        self.regret_sum = self.play_possitive_regrets.clone()
        self.regret_sum.add_(self.terminate_possitive_regrets)

        self.play_current_strategy.copy_(self.play_possitive_regrets)
        self.terminate_current_strategy.copy_(self.terminate_possitive_regrets)

        self.play_current_strategy.div_(self.regret_sum)
        self.terminate_current_strategy.div_(self.regret_sum)
        
        # 4.0 for poker, the range size is larger than the allowed hands
        # we need to make sure reconstruction does not choose a range
        # that is not allowed
        self.play_current_strategy.mul_(self.range_mask)
        self.terminate_current_strategy.mul_(self.range_mask)

        self.input_opponent_range = self.play_current_strategy.clone()  

        return self.input_opponent_range
