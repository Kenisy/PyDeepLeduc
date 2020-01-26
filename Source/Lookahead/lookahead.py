''' A depth-limited lookahead of the game tree used for re-solving.
@classmod lookahead'''

from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Lookahead.lookahead_builder import LookaheadBuilder
from Source.TerminalEquity.terminal_equity import TerminalEquity
from Source.Lookahead.cfrd_gadget import CFRDGadget
import torch

class Lookahead:
    def __init__(self):
        super().__init__()
        self.builder = LookaheadBuilder(self)

    def build_lookahead(self, tree):
        ''' Constructs the lookahead from a game's public tree.
        # 
        Must be called to initialize the lookahead.
        @param tree a public tree'''
        self.builder.build_from_tree(tree)

        self.terminal_equity = TerminalEquity()
        self.terminal_equity.set_board(tree.board)

    def resolve_first_node(self, player_range, opponent_range):
        ''' Re-solves the lookahead using input ranges.
        # 
        Uses the input range for the opponent instead of a gadget range, so only
        appropriate for re-solving the root node of the game tree (where ranges 
        are fixed).
        # 
        @{build_lookahead} must be called first.
        # 
        @param player_range a range vector for the re-solving player
        @param opponent_range a range vector for the opponent'''
        self.ranges_data[0][:, :, :, 0, :].copy_(player_range)  
        self.ranges_data[0][:, :, :, 1, :].copy_(opponent_range)  
        self._compute()

    def resolve(self, player_range, opponent_cfvs):
        ''' Re-solves the lookahead using an input range for the player and
        the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.

        @{build_lookahead} must be called first.

        @param player_range a range vector for the re-solving player
        @param opponent_cfvs a vector of cfvs achieved by the opponent
        before re-solving'''
        assert(player_range)
        assert(opponent_cfvs)
        
        self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)
        
        self.ranges_data[0][:, :, :, 0, :].copy_(player_range)
        self.reconstruction_opponent_cfvs = opponent_cfvs
        self._compute()

    def _compute(self):
        ''' Re-solves the lookahead.
        @local'''
        # 1.0 main loop
        for i in range(arguments.cfr_iters):
            self._set_opponent_starting_range(i)
            self._compute_current_strategies()
            self._compute_ranges()
            self._compute_update_average_strategies(i)
            self._compute_terminal_equities()   
            self._compute_cfvs()
            self._compute_regrets()
            self._compute_cumulate_average_cfvs(i)

        # 2.0 at the end normalize average strategy
        self._compute_normalize_average_strategies()
        # 2.1 normalize root's CFVs
        self._compute_normalize_average_cfvs()

    def _compute_current_strategies(self):
        ''' Uses regret matching to generate the players' current strategies.
        @local'''
        for d in range(1, self.depth):
            self.positive_regrets_data[d].copy_(self.regrets_data[d])
            self.positive_regrets_data[d].clamp_(self.regret_epsilon, tools.max_number())

            # 1.0 set regret of empty actions to 0
            self.positive_regrets_data[d].mul_(self.empty_action_mask[d])

            # 1.1  regret matching
            # note that the regrets as well as the CFVs have switched player indexing
            # TODO recheck
            self.regrets_sum[d] = torch.sum(self.positive_regrets_data[d], 1)
            player_current_strategy = self.current_strategy_data[d]
            player_regrets = self.positive_regrets_data[d]
            player_regrets_sum = self.regrets_sum[d]

            player_current_strategy.div_(player_regrets, player_regrets_sum.expand_as(player_regrets))

    def _compute_ranges(self):
        ''' Using the players' current strategies, computes their probabilities of
        reaching each state of the lookahead.
        @local'''
        for d in range(self.depth-1):
            current_level_ranges = self.ranges_data[d]
            next_level_ranges = self.ranges_data[d+1]

            prev_layer_terminal_actions_count = self.terminal_actions_count[d-1]
            prev_layer_actions_count = self.actions_count[d-1]
            prev_layer_bets_count = self.bets_count[d-1]
            gp_layer_nonallin_bets_count = self.nonallinbets_count[d-2]
            gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]


            # copy the ranges of inner nodes and transpose
            # TODO recheck
            self.inner_nodes[d].copy_(current_level_ranges[prev_layer_terminal_actions_count: -1, : gp_layer_nonallin_bets_count, :, :, :].transpose(2,3))

            super_view = self.inner_nodes[d]
            super_view = super_view.view(1, prev_layer_bets_count, -1, constants.players_count, game_settings.card_count)

            super_view = super_view.expand_as(next_level_ranges)
            next_level_strategies = self.current_strategy_data[d+1]

            next_level_ranges.copy_(super_view)

            # multiply the ranges of the acting player by his strategy
            next_level_ranges[:, :, :, self.acting_player[d], :].mul_(next_level_strategies)

    def _compute_update_average_strategies(self, _iter):
        ''' Updates the players' average strategies with their current strategies.
        @param iter the current iteration number of re-solving
        @local'''
        if _iter > arguments.cfr_skip_iters:
            # no need to go through layers since we care for the average strategy only in the first node anyway
            # note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
            self.average_strategies_data[1].add_(self.current_strategy_data[1])

    def _compute_terminal_equities_terminal_equity(self):
        ''' Using the players' reach probabilities, computes their counterfactual
        values at each lookahead state which is a terminal state of the game.
        @local'''
        for d in range(1, self.depth):

            # call term eq evaluation
            if self.tree.street == 1:
                if d > 1 or self.first_call_terminal != None:
                    self.terminal_equity.call_value(self.ranges_data[d][1][-1].view(-1, game_settings.card_count), self.cfvs_data[d][1][-1].view(-1, game_settings.card_count)) 
            else:
                assert(self.tree.street == 2)
                # on river, any call is terminal 
                if d > 1 or self.first_call_terminal != None:        
                    self.terminal_equity.call_value(self.ranges_data[d][1].view(-1, game_settings.card_count), self.cfvs_data[d][1].view(-1, game_settings.card_count))

            # folds
            self.terminal_equity.fold_value(self.ranges_data[d][0].view(-1, game_settings.card_count), self.cfvs_data[d][0].view(-1, game_settings.card_count))  

            # correctly set the folded player by mutliplying by -1
            fold_mutliplier = (self.acting_player[d]*2 - 3)
            self.cfvs_data[d][0, :, :, 0, :].mul_(fold_mutliplier)
            self.cfvs_data[d][0, :, :, 1, :].mul_(-fold_mutliplier)

    def _compute_terminal_equities_next_street_box(self):
        ''' Using the players' reach probabilities, calls the neural net to compute the
        players' counterfactual values at the depth-limited states of the lookahead.
        @local'''
        assert(self.tree.street == 1)

        for d in range(1, self.depth):
            
            if d > 1 or self.first_call_transition != None:
                if self.next_street_boxes_inputs == None:
                    self.next_street_boxes_inputs = {}
                if self.next_street_boxes_outputs == None:
                    self.next_street_boxes_outputs = {}
                
                if self.next_street_boxes_inputs[d] == None: 
                    self.next_street_boxes_inputs[d] = self.ranges_data[d][1, :, :, :, :].view(-1, constants.players_count, game_settings.card_count).clone().fill_(0)
                if self.next_street_boxes_outputs[d] == None: 
                    self.next_street_boxes_outputs[d] = self.next_street_boxes_inputs[d].clone()
                
                # now the neural net accepts the input for P1 and P2 respectively, so we need to swap the ranges if necessary
                self.next_street_boxes_outputs[d].copy_(self.ranges_data[d][1, :, :, :, :])
                
                if self.tree.current_player == 0:
                    self.next_street_boxes_inputs[d].copy_(self.next_street_boxes_outputs[d])
                else:
                    self.next_street_boxes_inputs[d][:, 0, :].copy_(self.next_street_boxes_outputs[d][:, 1, :])
                    self.next_street_boxes_inputs[d][:, 1, :].copy_(self.next_street_boxes_outputs[d][:, 0, :])
                
                self.next_street_boxes[d].get_value(self.next_street_boxes_inputs[d], self.next_street_boxes_outputs[d])      
                
                # now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
                if self.tree.current_player == 1:
                    self.next_street_boxes_inputs[d].copy_(self.next_street_boxes_outputs[d])
                    
                    self.next_street_boxes_outputs[d][:, 0, :].copy_(self.next_street_boxes_inputs[d][:, 1, :])
                    self.next_street_boxes_outputs[d][:, 1, :].copy_(self.next_street_boxes_inputs[d][:, 0, :])
                
                self.cfvs_data[d][1, :, :, :, :].copy_(self.next_street_boxes_outputs[d])

    def get_chance_action_cfv(self, action_index, board):
        ''' Gives the average counterfactual values for the opponent during re-solving 
        after a chance event (the betting round changes and more cards are dealt).

        Used during continual re-solving to track opponent cfvs. The lookahead must 
        first be re-solved with @{resolve} or @{resolve_first_node}.

        @param action_index the action taken by the re-solving player at the start
        of the lookahead
        @param board a tensor of board cards, updated by the chance event
        @return a vector of cfvs'''
        box_outputs = None
        next_street_box = None
        batch_index = None
        pot_mult = None
        
        assert(not ((action_index == 2) and (self.first_call_terminal)))
        
        # check if we should not use the first layer for transition call
        if action_index == 2 and self.first_call_transition != None:
            box_outputs = self.next_street_boxes_inputs[1].clone().fill_(0)
            assert(box_outputs.size(0) == 1)
            batch_index = 1
            next_street_box = self.next_street_boxes[1]
            pot_mult = self.pot_size[1][1]
        else:
            batch_index = action_index - 1 # remove fold
            if self.first_call_transition:
                batch_index = batch_index - 1
        
            box_outputs = self.next_street_boxes_inputs[2].clone().fill_(0)
            next_street_box = self.next_street_boxes[2]
            pot_mult = self.pot_size[2][1]
        
        
        if box_outputs == None:
            assert(False)
        next_street_box.get_value_on_board(board, box_outputs)
        
        box_outputs.mul_(pot_mult)
        
        out = box_outputs[batch_index][1-self.tree.current_player]
        return out

    def _compute_terminal_equities(self):
        ''' Using the players' reach probabilities, computes their counterfactual
        values at all terminal states of the lookahead.

        These include terminal states of the game and depth-limited states.
        @local'''
        if self.tree.street == 1:
            self._compute_terminal_equities_next_street_box()

        self._compute_terminal_equities_terminal_equity() 

        # multiply by pot scale factor
        for d in range(1, self.depth):
            self.cfvs_data[d].mul_(self.pot_size[d])

    def _compute_cfvs(self):
        ''' Using the players' reach probabilities and terminal counterfactual
        values, computes their cfvs at all states of the lookahead.
        @local'''
        for d in range(self.depth-1, 1, -1):
            gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
            ggp_layer_nonallin_bets_count = self.nonallinbets_count[d-3]

            self.cfvs_data[d][:, :, :, 0, :].mul_(self.empty_action_mask[d])
            self.cfvs_data[d][:, :, :, 1, :].mul_(self.empty_action_mask[d])

            self.placeholder_data[d].copy_(self.cfvs_data[d])

            # player indexing is swapped for cfvs
            self.placeholder_data[d][:, :, :, self.acting_player[d], :].mul_(self.current_strategy_data[d])

            # TODO recheck
            self.regrets_sum[d] = torch.sum(self.placeholder_data[d], 1)

            # use a swap placeholder to change {{1,2,3}, {4,5,6}} into {{1,2}, {3,4}, {5,6}}
            swap = self.swap_data[d-1]
            swap.copy_(self.regrets_sum[d])
            self.cfvs_data[d-1][gp_layer_terminal_actions_count: -1, : ggp_layer_nonallin_bets_count, :, :, :].copy_(swap.transpose(2,3))

    def _compute_cumulate_average_cfvs(self, _iter):
        ''' Updates the players' average counterfactual values with their cfvs from the
        current iteration.
        @param iter the current iteration number of re-solving
        @local'''
        if _iter > arguments.cfr_skip_iters:
            self.average_cfvs_data[0].add_(self.cfvs_data[0])
            
            self.average_cfvs_data[1].add_(self.cfvs_data[1])

    def _compute_normalize_average_strategies(self):
        ''' Normalizes the players' average strategies.

        Used at the end of re-solving so that we can track un-normalized average
        strategies, which are simpler to compute.
        @local'''
        # using regrets_sum as a placeholder container
        player_avg_strategy = self.average_strategies_data[1]
        player_avg_strategy_sum = self.regrets_sum[1]

        # TODO recheck
        player_avg_strategy_sum = torch.sum(player_avg_strategy, 1)
        player_avg_strategy.div_(player_avg_strategy_sum.expand_as(player_avg_strategy))
        
        # if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
        # it sums to one -> now we set to always fold
        player_avg_strategy[0][player_avg_strategy[0].ne(player_avg_strategy[0])] = 1
        player_avg_strategy[player_avg_strategy.ne(player_avg_strategy)] = 0

    def _compute_normalize_average_cfvs(self):
        ''' Normalizes the players' average counterfactual values.

        Used at the end of re-solving so that we can track un-normalized average
        cfvs, which are simpler to compute.
        @local'''
        self.average_cfvs_data[0].div(arguments.cfr_iters - arguments.cfr_skip_iters)

    def _compute_regrets(self):
        ''' Using the players' counterfactual values, updates their total regrets
        for every state in the lookahead.
        @local'''
        for d in range(self.depth-1, 1, -1):
            gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
            gp_layer_bets_count = self.bets_count[d-2]
            ggp_layer_nonallin_bets_count = self.nonallinbets_count[d-3]

            current_regrets = self.current_regrets_data[d]
            current_regrets.copy_(self.cfvs_data[d][:, :, :, self.acting_player[d], :])

            next_level_cfvs = self.cfvs_data[d-1]

            parent_inner_nodes = self.inner_nodes_p1[d-1]
            parent_inner_nodes.copy_(next_level_cfvs[gp_layer_terminal_actions_count : -1, : ggp_layer_nonallin_bets_count, :, self.acting_player[d], :].transpose(2,3))
            parent_inner_nodes = parent_inner_nodes.view(1, gp_layer_bets_count, -1, game_settings.card_count)
            parent_inner_nodes = parent_inner_nodes.expand_as(current_regrets)

            current_regrets.sub_(parent_inner_nodes)
            
            self.regrets_data[d].add_(self.regrets_data[d], current_regrets)

            # (CFR+)
            self.regrets_data[d].clamp_(0,  tools.max_number())


    def get_results(self):
        ''' Gets the results of re-solving the lookahead.

        The lookahead must first be re-solved with @{resolve} or 
        @{resolve_first_node}.

        @return a table containing the fields.

        * `strategy`. an AxK tensor containing the re-solve player's strategy at the
        root of the lookahead, where A is the number of actions and K is the range size

        * `achieved_cfvs`. a vector of the opponent's average counterfactual values at the 
        root of the lookahead

        * `children_cfvs`. an AxK tensor of opponent average counterfactual values after
        each action that the re-solve player can take at the root of the lookahead'''
        out = {}
        
        actions_count = self.average_strategies_data[1].size(0)

        # 1.0 average strategy
        # [actions x range]
        # lookahead already computes the averate strategy we just convert the dimensions
        out.strategy = self.average_strategies_data[1].view(-1, game_settings.card_count).clone()

        # 2.0 achieved opponent's CFVs at the starting node  
        out.achieved_cfvs = self.average_cfvs_data[0].view(constants.players_count, game_settings.card_count)[0].clone()
        
        # 3.0 CFVs for the acting player only when resolving first node
        if self.reconstruction_opponent_cfvs != None:
            out.root_cfvs = None
        else:
            out.root_cfvs = self.average_cfvs_data[0].view(constants.players_count, game_settings.card_count)[1].clone()
            
            # swap cfvs indexing
            out.root_cfvs_both_players = self.average_cfvs_data[0].view(constants.players_count, game_settings.card_count).clone()
            out.root_cfvs_both_players[1].copy_(self.average_cfvs_data[0].view(constants.players_count, game_settings.card_count)[0])
            out.root_cfvs_both_players[0].copy_(self.average_cfvs_data[0].view(constants.players_count, game_settings.card_count)[1])
        
        # 4.0 children CFVs
        # [actions x range]
        out.children_cfvs = self.average_cfvs_data[1][:, :, :, 0, :].clone().view(-1, game_settings.card_count)

        # IMPORTANT divide average CFVs by average strategy in here
        scaler = self.average_strategies_data[1].view(-1, game_settings.card_count).clone()

        range_mul = self.ranges_data[0][:, :, :, 0, :].view(1, game_settings.card_count).clone()
        range_mul = range_mul.expand_as(scaler)

        scaler = scaler.mul(range_mul)
        scaler = scaler.sum(dim=1, keepdim=True).expand_as(range_mul).clone()
        scaler = scaler.mul(arguments.cfr_iters - arguments.cfr_skip_iters)   
        
        out.children_cfvs.div_(scaler)  
        
        assert(out.strategy)
        assert(out.achieved_cfvs)
        assert(out.children_cfvs)
        
        return out

    def _set_opponent_starting_range(self, iteration):
        ''' Generates the opponent's range for the current re-solve iteration using
        the @{cfrd_gadget|CFRDGadget}.
        @param iteration the current iteration number of re-solving
        @local'''
        if self.reconstruction_opponent_cfvs:
            # note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
            opponent_range = self.reconstruction_gadget.compute_opponent_range(self.cfvs_data[0][:, :, :, 0, :], iteration)
            self.ranges_data[0][:, :, :, 1, :].copy_(opponent_range)