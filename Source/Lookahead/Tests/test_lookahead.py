import sys
sys.path.append(sys.path[0] + '/../../../')
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Game.card_tools import card_tools
from Source.Game.card_to_string_conversion import card_to_string
from Source.Lookahead.resolving import Resolving
from Source.Tree.tree_builder import TreeNode
import torch

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    resolving = Resolving()
    current_node = TreeNode()

    current_node.board = card_to_string.string_to_board('Ks')
    current_node.street = 2
    current_node.current_player = constants.players.P1
    current_node.bets = arguments.Tensor([100, 100])

    player_range = card_tools.get_random_range(current_node.board, 2)
    opponent_range = card_tools.get_random_range(current_node.board, 4)

    resolving.resolve_first_node(current_node, player_range, opponent_range)

    # result = resolving.resolve(current_node, player_range, opponent_range)
    print(resolving.get_root_cfv_both_players())