import sys
sys.path.append(sys.path[0] + '/../../../')
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Tree.tree_builder import *
from Source.Game.card_to_string_conversion import card_to_string

if __name__ == "__main__":
    builder = PokerTreeBuilder()
    params = TreeParams()
    params.root_node = TreeNode()
    params.root_node.board = card_to_string.string_to_board('')
    params.root_node.street = 1
    params.root_node.current_player = constants.players.P1
    params.root_node.bets = arguments.Tensor([100, 100])

    tree = builder.build_tree(params)