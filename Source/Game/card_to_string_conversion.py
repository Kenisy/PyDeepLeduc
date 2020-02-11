''' Converts between string and numeric representations of cards.
@module card_to_string_conversion '''

from Source.Settings.game_settings import game_settings
from Source.Settings.arguments import arguments
import torch

class M:
    def __init__(self):
        super().__init__()
        # All possible card suits - only the first 2 are used in Leduc Hold'em.
        self.suit_table = ['h', 's', 'c', 'd']

        # All possible card ranks - only the first 3-4 are used in Leduc Hold'em and 
        # variants.
        self.rank_table = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

        # Holds the string representation for every possible card, indexed by its 
        # numeric representation.
        self.card_to_string_table = {}
        for card in range(game_settings.card_count): 
            rank_name = self.rank_table[self.card_to_rank(card)]
            suit_name = self.suit_table[self.card_to_suit(card)]
            self.card_to_string_table[card] =  rank_name + suit_name

        # Holds the numeric representation for every possible card, indexed by its 
        # string representation.
        self.string_to_card_table = {}
        for card in range(game_settings.card_count): 
            self.string_to_card_table[self.card_to_string_table[card]] = card

    def card_to_suit(self, card):
        ''' Gets the suit of a card.
        @param card the numeric representation of the card
        @return the index of the suit'''
        return card % game_settings.suit_count

    def card_to_rank(self, card):
        ''' Gets the rank of a card.
        @param card the numeric representation of the card
        @return the index of the rank'''
        return card // game_settings.suit_count
 
    def card_to_string(self, card):
        ''' Converts a card's numeric representation to its string representation.
        @param card the numeric representation of a card
        @return the string representation of the card'''
        assert card >= 0 and card < game_settings.card_count
        return self.card_to_string_table[int(card)]

    def cards_to_string(self, cards):
        ''' Converts several cards' numeric representations to their string 
        representations.
        @param cards a vector of numeric representations of cards
        @return a string containing each card's string representation, concatenated'''
        if cards.dim() == 0:
            return ""
        
        out = ""
        for card in range(cards.size(0)):
            out = out + self.card_to_string(cards[card])
        return out

    def string_to_card(self, card_string):
        ''' Converts a card's string representation to its numeric representation.
        @param card_string the string representation of a card
        @return the numeric representation of the card'''
        card = self.string_to_card_table[card_string]
        assert card >= 0 and card < game_settings.card_count
        return card

    def string_to_board(self, card_string):
        ''' Converts a string representing zero or one board cards to a 
        vector of numeric representations.
        @param card_string either the empty string or a string representation of a 
        card
        @return either an empty tensor or a tensor containing the numeric 
        representation of the card'''
        # assert card_string
        
        if card_string == '':
            return arguments.IntTensor()
        
        return arguments.IntTensor([self.string_to_card(card_string)])

card_to_string = M()