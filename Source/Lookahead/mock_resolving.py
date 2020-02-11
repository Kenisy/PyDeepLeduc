''' Implements the re-solving interface used by @{resolving} with functions
that do nothing.

Used for debugging.
@classmod mock_resolving'''

from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings

class MockResolving:
    def resolve_first_node(self, node, player_range, opponent_range):
        ''' Does nothing.

        Params:
            node: the node to "re-solve"
            player_range [opt]: not used
            opponent_range [opt]: not used
        '''
        self.node = node
        self.action_count = self.node.actions.size(0)

    def resolve(self, node, player_range, opponent_cfvs):
        ''' Does nothing.

        Params:
            node the node to "re-solve"
            player_range [opt]: not used
            opponent_cfvs [opt]: not used
        '''
        self.node = node
        self.action_count = self.node.actions.size(0)

    def get_possible_actions(self):
        ''' Gives the possible actions at the re-solve node.
        Return the actions that can be taken at the re-solve node
        '''
        return self.node.actions

    def get_root_cfv(self):
        ''' Returns an arbitrary vector.
        Return a vector of 1s
        '''
        return arguments.Tensor(game_settings.card_count).fill_(1)

    def get_action_cfv(self, action):
        ''' Returns an arbitrary vector.

        Params:
            action [opt]: not used
        Return a vector of 1s
        '''
        return arguments.Tensor(game_settings.card_count).fill_(1)

    def get_chance_action_cfv(self, player_action, board):
        ''' Returns an arbitrary vector.

        Params:
            player_action [opt]: not used
            board [opt]: not used
        Return a vector of 1s
        '''
        return arguments.Tensor(game_settings.card_count).fill_(1)

    def get_action_strategy(self, action):
        ''' Returns an arbitrary vector.

        Params:
            action [opt]: not used
        Return a vector of 1s
        '''
        return arguments.Tensor(game_settings.card_count).fill_(1)