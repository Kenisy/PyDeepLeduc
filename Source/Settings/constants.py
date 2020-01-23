class constants:
    # the number of players in the game
    players_count = 2
    # the number of betting rounds in the game
    streets_count = 2

    # IDs for each player and chance
    # @field chance `0`
    # @field P1 `1`
    # @field P2 `2`
    class players:
        chance = -1
        P1 = 0
        P2 = 1
    players = players()

    # IDs for terminal nodes (either after a fold or call action) and nodes that follow a check action
    # @field terminal_fold (terminal node following fold) `-2`
    # @field terminal_call (terminal node following call) `-1`
    # @field chance_node (node for the chance player) `0`
    # @field check (node following check) `-1`
    # @field inner_node (any other node) `1`
    class node_types:
        terminal_fold = -2
        terminal_call = -1
        check = -1
        chance_node = 0
        inner_node = 1
    node_types = node_types()

    # IDs for fold and check/call actions
    # @field fold `-2`
    # @field ccall (check/call) `-1`
    class actions:
        fold = -2
        ccall = -1
    actions = actions()

constants = constants()