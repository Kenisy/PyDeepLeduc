''' Builds a public tree for Leduc Hold'em or variants.

Each node of the tree contains the following fields.

* `node_type`. an element of @{constants.node_types} (if applicable)

* `street`. the current betting round

* `board`. a possibly empty vector of board cards

* `board_string`. a string representation of the board cards

* `current_player`. the player acting at the node

* `bets`. the number of chips that each player has committed to the pot
# 
* `pot`. half the pot size, equal to the smaller number in `bets`
# 
* `children`. a list of children nodes
@classmod tree_builder
'''
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Game.card_tools import card_tools
from Source.Game.card_to_string_conversion import card_to_string
from Source.Tree.strategy_filling import StrategyFilling
from Source.Game.bet_sizing import BetSizing

class TreeNode:
    def __init__(self):
        super().__init__()
        self.type = None
        self.node_type = None
        self.street = None
        self.board = None
        self.board_string = None
        self.current_player = None
        self.bets = None
        self.terminal = None
        # visualiser
        self.margin = None
        self.cfv_infset = None
        self.cfv_br_infset = None
        self.epsilon = None
        self.lookahead_coordinates = None
        self.ranges_absolute = None
        self.cf_values = None
        self.cf_values_br = None
        # cfr
        self.regrets = None
        self.possitive_regrets = None
        self.iter_weight_sum = None

class TreeParams:
    def __init__(self):
        super().__init__()
        self.root_node = None
        self.bet_sizing = None
        self.limit_to_street = None

class PokerTreeBuilder:
    
    def _get_children_nodes_transition_call(self, parent_node):
        ''' Creates the child node after a call which transitions between betting 
        rounds.

        Params:
            parent_node: the node at which the transition call happens
        Return a list containing the child node
        '''
        chance_node = TreeNode()
        chance_node.node_type = constants.node_types.chance_node
        chance_node.street = parent_node.street
        chance_node.board = parent_node.board
        chance_node.board_string = parent_node.board_string
        chance_node.current_player = constants.players.chance  
        chance_node.bets = parent_node.bets.clone()

        return [chance_node]

    def _get_children_nodes_chance_node(self, parent_node):
        ''' Creates the children nodes after a chance node.

        Params:
            parent_node: the chance node
        Return a list of children nodes
        '''
        assert parent_node.current_player == constants.players.chance
        
        if self.limit_to_street:
            return []

        next_boards = card_tools.get_second_round_boards()
        next_boards_count = next_boards.size(0)

        subtree_height = -1
        children = []

        # 1.0 iterate over the next possible boards to build the corresponding subtrees
        for i in range(next_boards_count):
            next_board = next_boards[i]
            next_board_string = card_to_string.cards_to_string(next_board)

            child = TreeNode()

            child.node_type = constants.node_types.inner_node
            child.parent = parent_node
            child.current_player = constants.players.P1
            child.street = parent_node.street + 1
            child.board = next_board
            child.board_string = next_board_string
            child.bets = parent_node.bets.clone()

            children.append(child)

        return children

    def _fill_additional_attributes(self, node):
        ''' Fills in additional convenience attributes which only depend on existing
        node attributes.

        Params:
            node: the node
        '''
        node.pot = node.bets.min()

    def _get_children_player_node(self, parent_node):
        ''' Creates the children nodes after a player node.

        Params:
            parent_node: the chance node
        Return a list of children nodes
        '''
        assert parent_node.current_player != constants.players.chance

        children = []
        
        # 1.0 fold action
        fold_node = TreeNode()
        fold_node.type = constants.node_types.terminal_fold
        fold_node.terminal = True
        fold_node.current_player = 1 - parent_node.current_player
        fold_node.street = parent_node.street 
        fold_node.board = parent_node.board
        fold_node.board_string = parent_node.board_string
        fold_node.bets = parent_node.bets.clone()
        children.append(fold_node)
        
        # 2.0 check action
        if parent_node.current_player == constants.players.P1 and (parent_node.bets[0] == parent_node.bets[1]):
            check_node = TreeNode()
            check_node.type = constants.node_types.check
            check_node.terminal = False
            check_node.current_player = 1 - parent_node.current_player
            check_node.street = parent_node.street 
            check_node.board = parent_node.board
            check_node.board_string = parent_node.board_string
            check_node.bets = parent_node.bets.clone()
            children.append(check_node)
        # transition call
        elif parent_node.street == 1 and ( (parent_node.current_player == constants.players.P2 and parent_node.bets[0] == parent_node.bets[1]) \
            or (parent_node.bets[0] != parent_node.bets[1] and parent_node.bets.max() < arguments.stack) ): 
            chance_node = TreeNode()
            chance_node.node_type = constants.node_types.chance_node
            chance_node.street = parent_node.street
            chance_node.board = parent_node.board
            chance_node.board_string = parent_node.board_string
            chance_node.current_player = constants.players.chance  
            chance_node.bets = parent_node.bets.clone().fill_(parent_node.bets.max())
            children.append(chance_node)
        else:
        # 2.0 terminal call - either last street or allin
            terminal_call_node = TreeNode()
            terminal_call_node.type = constants.node_types.terminal_call
            terminal_call_node.terminal = True
            terminal_call_node.current_player = 1 - parent_node.current_player
            terminal_call_node.street = parent_node.street 
            terminal_call_node.board = parent_node.board
            terminal_call_node.board_string = parent_node.board_string
            terminal_call_node.bets = parent_node.bets.clone().fill_(parent_node.bets.max())
            children.append(terminal_call_node)

        # 3.0 bet actions    
        possible_bets = self.bet_sizing.get_possible_bets(parent_node)
        
        if possible_bets.dim() != 0 and possible_bets.size(0) > 0:
            assert possible_bets.size(1) == 2
            for i in range(possible_bets.size(0)):
                child = TreeNode()
                child.parent = parent_node
                child.current_player = 1 - parent_node.current_player
                child.street = parent_node.street 
                child.board = parent_node.board
                child.board_string = parent_node.board_string
                child.bets = possible_bets[i]
                children.append(child)
        
        return children

    def _get_children_nodes(self, parent_node):
        ''' Creates the children after a node.

        Params:
            parent_node: the node to create children for
        Return a list of children nodes
        '''
        # is this a transition call node (leading to a chance node)?
        call_is_transit = parent_node.current_player == constants.players.P2 and parent_node.bets[0] == parent_node.bets[1] and parent_node.street < constants.streets_count
        
        chance_node = parent_node.current_player == constants.players.chance
        # transition call -> create a chance node
        if parent_node.terminal:
            return []
        # chance node
        elif chance_node:
            return self._get_children_nodes_chance_node(parent_node)
        # inner nodes -> handle bet sizes
        else:
            return self._get_children_player_node(parent_node)

        assert False

    def _build_tree_dfs(self, current_node):
        ''' Recursively build the (sub)tree rooted at the current node.

        Params:
            current_node: the root to build the (sub)tree from
        Return `current_node` after the (sub)tree has been built
        '''
        self._fill_additional_attributes(current_node)
        children = self._get_children_nodes(current_node)
        current_node.children = children
        
        depth = 0

        current_node.actions = arguments.Tensor(len(children))
        for i in range(len(children)):
            children[i].parent = current_node
            self._build_tree_dfs(children[i])
            depth = max(depth, children[i].depth)
            
            if i == 0:
                current_node.actions[i] = constants.actions.fold
            elif i == 1:
                current_node.actions[i] = constants.actions.ccall
            else:  
                current_node.actions[i] = children[i].bets.max()
        
        current_node.depth = depth + 1
        
        return current_node

    def build_tree(self, params):
        ''' Builds the tree.

        Params:
            params: table of tree parameters, containing the following fields:
                * `street`: the betting round of the root node
                * `bets`: the number of chips committed at the root node by each player
                * `current_player`: the acting player at the root node
                * `board`: a possibly empty vector of board cards at the root node
                * `limit_to_street`: if `true`, only build the current betting round
                * `bet_sizing` (optional): a @{bet_sizing} object which gives the allowed
                    bets for each player 
        Return the root node of the built tree'''
        root = TreeNode()
        # copy necessary stuff from the root_node not to touch the input
        root.street = params.root_node.street
        root.bets = params.root_node.bets.clone()
        root.current_player = params.root_node.current_player
        root.board = params.root_node.board.clone()
        
        if not params.bet_sizing:
            params.bet_sizing = BetSizing(arguments.Tensor(arguments.bet_sizing))

        assert params.bet_sizing

        self.bet_sizing = params.bet_sizing
        self.limit_to_street = params.limit_to_street

        self._build_tree_dfs(root)
        
        strategy_filling = StrategyFilling()
        strategy_filling.fill_uniform(root)
        
        return root
