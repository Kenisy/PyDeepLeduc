import sys
sys.path.append(sys.path[0] + '/../../../')
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Tree.tree_builder import *
from Source.Game.card_to_string_conversion import card_to_string
from Source.Tree.tree_values import TreeValues
from Source.Tree.tree_strategy_filling import TreeStrategyFilling
from Source.Tree.tree_visualiser import TreeVisualiser

if __name__ == "__main__":
    builder = PokerTreeBuilder()
    params = TreeParams()
    params.root_node = TreeNode()
    params.root_node.board = card_to_string.string_to_board('')
    params.root_node.street = 1
    params.root_node.current_player = constants.players.P1
    params.root_node.bets = arguments.Tensor([100, 100])

    tree = builder.build_tree(params)

    filling = TreeStrategyFilling()

    range1 = card_tools.get_uniform_range(params.root_node.board)
    range2 = card_tools.get_uniform_range(params.root_node.board)

    filling.fill_strategies(tree, 0, range1, range2)
    filling.fill_strategies(tree, 1, range1, range2)

    starting_ranges = arguments.Tensor(constants.players_count, game_settings.card_count)
    starting_ranges[0].copy_(range1)
    starting_ranges[1].copy_(range2)

    tree_values = TreeValues()
    tree_values.compute_values(tree, starting_ranges)

    print('Exploitability: ' + str(tree.exploitability.item()) + '[chips]' )

    visualiser = TreeVisualiser()
    visualiser.graphviz(tree, "tree_strategy_filling")
