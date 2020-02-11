''' Samples random card combinations.'''
from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
import numpy as np

class M:

    def generate_cards(self, count ):
        ''' Samples a random set of cards.

        Each subset of the deck of the correct size is sampled with 
        uniform probability.
        
        Params:
            count: the number of cards to sample
        Return a vector of cards, represented numerically'''
        # marking all used cards
        used_cards = arguments.IntTensor(game_settings.card_count).zero_()
        
        out = arguments.IntTensor(count)
        # counter for generated cards
        generated_cards_count = 0
        while(generated_cards_count < count):
            card = np.random.randint(0, game_settings.card_count)
            if ( used_cards[card] == 0 ): 
                out[generated_cards_count] = card
                used_cards[card] = 1
                generated_cards_count = generated_cards_count + 1
        return out

card_generator = M()
