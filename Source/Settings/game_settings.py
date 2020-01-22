class M:
    # the number of card suits in the deck
    suit_count = 2
    # the number of card ranks in the deck
    rank_count = 3
    # the total number of cards in the deck
    card_count = suit_count * rank_count
    # the number of public cards dealt in the game (revealed after the first
    # betting round)
    board_card_count = 1
    # the number of players in the game
    player_count = 2

game_settings = M()