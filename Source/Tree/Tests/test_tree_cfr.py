import sys
sys.path.append(sys.path[0] + '/../../../')
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Tree.tree_builder import *
from Source.Game.card_to_string_conversion import card_to_string
from Source.Tree.tree_cfr import TreeCFR
from Source.Tree.tree_values import TreeValues

if __name__ == "__main__":
    builder = PokerTreeBuilder()
    params = TreeParams()
    params.root_node = TreeNode()
    params.root_node.board = card_to_string.string_to_board('')
    params.root_node.street = 1
    params.root_node.current_player = constants.players.P1
    params.root_node.bets = arguments.Tensor([100, 100])

    tree = builder.build_tree(params)

    starting_ranges = arguments.Tensor(constants.players_count, game_settings.card_count)

    starting_ranges[0].copy_(card_tools.get_uniform_range(params.root_node.board))
    starting_ranges[1].copy_(card_tools.get_uniform_range(params.root_node.board))

    tree_cfr = TreeCFR()
    tree_cfr.run_cfr(tree, starting_ranges)

    tree_values = TreeValues()
    tree_values.compute_values(tree, starting_ranges)

    print('Exploitability: ' + str(tree.exploitability.item()) + '[chips]' )