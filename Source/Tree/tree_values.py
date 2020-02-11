''' Computes the expected value of a strategy profile on a game's public tree,
as well as the value of a best response against the profile.
'''
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Game.card_tools import card_tools
from Source.TerminalEquity.terminal_equity import TerminalEquity
import torch

class TreeValues:
    def __init__(self):
        super().__init__()
        self.terminal_equity = TerminalEquity()

    def _fill_ranges_dfs(self, node, ranges_absolute):
        ''' Recursively walk the tree and calculate the probability of reaching each
        node using the saved strategy profile.
        
        The reach probabilities are saved in the `ranges_absolute` field of each
        node.

        Params:
            node: the current node of the tree
            ranges_absolute: a 2xK tensor containing the probabilities of each 
                player reaching the current node with each private hand
        '''
        node.ranges_absolute = ranges_absolute.clone()

        if(node.terminal):
            return
        
        assert(node.strategy != None)

        actions_count = len(node.children) 
        
        # check that it's a legal strategy
        strategy_to_check = node.strategy
        
        hands_mask = card_tools.get_possible_hand_indexes(node.board)
        
        if node.current_player != constants.players.chance:
            checksum = strategy_to_check.sum(dim=0)
            assert(not torch.any(strategy_to_check.lt(0)))
            assert(not torch.any(checksum.gt(1.001)))    
            assert(not torch.any(checksum.lt(0.999)))
            assert(not torch.any(checksum.ne(checksum)))
        
        assert(node.ranges_absolute.lt(0).sum() == 0)
        assert(node.ranges_absolute.gt(1).sum() == 0)
        
        # check if the range consists only of cards that don't overlap with the board
        impossible_hands_mask = hands_mask.clone().fill_(1) - hands_mask
        impossible_range_sum = node.ranges_absolute.clone().mul(impossible_hands_mask.view(1, game_settings.card_count).expand_as(node.ranges_absolute)).sum()  
        assert impossible_range_sum == 0, impossible_range_sum
            
        children_ranges_absolute = arguments.Tensor(len(node.children), constants.players_count, game_settings.card_count)
        
        # chance player
        if node.current_player == constants.players.chance:
            # multiply ranges of both players by the chance prob
            children_ranges_absolute[:, constants.players.P1, :].copy_(node.ranges_absolute[constants.players.P1].repeat(actions_count, 1))
            children_ranges_absolute[:, constants.players.P2, :].copy_(node.ranges_absolute[constants.players.P2].repeat(actions_count, 1))
            
            children_ranges_absolute[:, constants.players.P1, :].mul_(node.strategy)
            children_ranges_absolute[:, constants.players.P2, :].mul_(node.strategy)
        # player
        else:
            # copy the range for the non-acting player  
            children_ranges_absolute[:, 1-node.current_player, :] = node.ranges_absolute[1-node.current_player].clone().repeat(actions_count, 1) 
            
            # multiply the range for the acting player using his strategy    
            ranges_mul_matrix = node.ranges_absolute[node.current_player].repeat(actions_count, 1) 
            children_ranges_absolute[:, node.current_player, :] = torch.mul(node.strategy, ranges_mul_matrix)
        
        # fill the ranges for the children
        for i in range(len(node.children)):
            child_node = node.children[i]
            child_range = children_ranges_absolute[i]
            
            # go deeper
            self._fill_ranges_dfs(child_node, child_range)

    def _compute_values_dfs(self, node):
        ''' Recursively calculate the counterfactual values for each player at each
        node of the tree using the saved strategy profile.
        
        The cfvs for each player in the given strategy profile when playing against
        each other is stored in the `cf_values` field for each node. The cfvs for
        a best response against each player in the profile are stored in the 
        `cf_values_br` field for each node.

        Params:
            node: the current node
        '''
        # compute values using terminal_equity in terminal nodes
        if(node.terminal):
        
            assert(node.type == constants.node_types.terminal_fold or node.type == constants.node_types.terminal_call)
        
            self.terminal_equity.set_board(node.board)    
            
            values = node.ranges_absolute.clone().fill_(0)

            if(node.type == constants.node_types.terminal_fold):
                self.terminal_equity.tree_node_fold_value(node.ranges_absolute, values, 1-node.current_player)
            else:
                self.terminal_equity.tree_node_call_value(node.ranges_absolute, values)

            # multiply by the pot
            values = values * node.pot

            node.cf_values = values.view_as(node.ranges_absolute)
            node.cf_values_br = values.view_as(node.ranges_absolute)
        else:

            actions_count = len(node.children)
            ranges_size = node.ranges_absolute.size(1)

            # [[actions, players, ranges]]
            cf_values_allactions = arguments.Tensor(len(node.children), 2, ranges_size).fill_(0)
            cf_values_br_allactions = arguments.Tensor(len(node.children), 2, ranges_size).fill_(0)

            for i in range(len(node.children)):    
                child_node = node.children[i]
                self._compute_values_dfs(child_node)
                cf_values_allactions[i] = child_node.cf_values
                cf_values_br_allactions[i] = child_node.cf_values_br

            node.cf_values = arguments.Tensor(2, ranges_size).fill_(0)
            node.cf_values_br = arguments.Tensor(2, ranges_size).fill_(0)

            # strategy = [[actions x range]]
            strategy_mul_matrix = node.strategy.view_as(arguments.Tensor(actions_count, ranges_size))

            # compute CFVs given the current strategy for this node
            if node.current_player == constants.players.chance:
                node.cf_values = cf_values_allactions.sum(dim=0)
                node.cf_values_br = cf_values_br_allactions.sum(dim=0)
            else:
                node.cf_values[node.current_player] = torch.mul(strategy_mul_matrix, cf_values_allactions[:, node.current_player, :]).sum(dim=0)
                node.cf_values[1-node.current_player] = (cf_values_allactions[:, 1-node.current_player, :]).sum(dim=0)
            
                # compute CFVs given the BR strategy for this node
                node.cf_values_br[1 - node.current_player] = cf_values_br_allactions[:, 1 - node.current_player, :].sum(dim=0)
                node.cf_values_br[node.current_player] = cf_values_br_allactions[:, node.current_player, :].max(dim=0).values

        # counterfactual values weighted by the reach prob
        node.cfv_infset = arguments.Tensor(2)
        node.cfv_infset[0] = node.cf_values[0] @ node.ranges_absolute[0]
        node.cfv_infset[1] = node.cf_values[1] @ node.ranges_absolute[1]

        # compute CFV-BR values weighted by the reach prob
        node.cfv_br_infset = arguments.Tensor(2)
        node.cfv_br_infset[0] = node.cf_values_br[0] @ node.ranges_absolute[0]
        node.cfv_br_infset[1] = node.cf_values_br[1] @ node.ranges_absolute[1]

        node.epsilon = node.cfv_br_infset - node.cfv_infset
        node.exploitability = node.epsilon.mean()

    def compute_values(self, root, starting_ranges=None ):
        ''' Compute the self play and best response values of a strategy profile on
        the given game tree.

        The cfvs for each player in the given strategy profile when playing against
        each other is stored in the `cf_values` field for each node. The cfvs for
        a best response against each player in the profile are stored in the 
        `cf_values_br` field for each node.

        Params:
            root: The root of the game tree. Each node of the tree is assumed to
                have a strategy saved in the `strategy` field.
            starting_ranges [opt]: probability vectors over player private hands
                at the root node (default uniform)'''
        # 1.0 set the starting range
        uniform_ranges = arguments.Tensor(constants.players_count, game_settings.card_count).fill_(1.0/game_settings.card_count)  
        if starting_ranges == None:
            starting_ranges = uniform_ranges
        
        # 2.0 check the starting ranges
        checksum = starting_ranges.sum(dim=1)
        assert abs(checksum[0] - 1) < 0.0001, 'starting range does not sum to 1'
        assert abs(checksum[1] - 1) < 0.0001, 'starting range does not sum to 1'
        assert(starting_ranges.lt(0).sum() == 0) 
        
        # 3.0 compute the values  
        self._fill_ranges_dfs(root, starting_ranges)
        self._compute_values_dfs(root)
