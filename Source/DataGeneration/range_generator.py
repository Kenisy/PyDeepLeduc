''' Samples random probability vectors for use as player ranges.'''

from Source.Settings.arguments import arguments
from Source.Game.Evaluation.evaluator import evaluator
from Source.Game.card_tools import card_tools
import torch
import numpy as np

class RangeGenerator:

    def _generate_recursion(self, cards, mass):
        ''' Recursively samples a section of the range vector.

        Params:
            cards: an NxJ section of the range tensor, where N is the batch size 
                and J is the length of the range sub-vector
            mass: a vector of remaining probability mass for each batch member '''
        batch_size = cards.size(0)
        assert(mass.size(0) == batch_size)
        # we terminate recursion at size of 1
        card_count = cards.size(1)
        if card_count == 1:
            cards[:, 0].copy_(mass) 
        else:
            rand = torch.rand(batch_size)
            mass1 = mass.clone().mul(rand)
            mass2 = mass -mass1
            halfSize = card_count // 2
            # if the tensor contains an odd number of cards, randomize which way the
            # middle card goes
            if card_count % 2 != 0:
                halfSize = halfSize + np.random.randint(0, 2)
            self._generate_recursion(cards[:, : halfSize], mass1)
            self._generate_recursion(cards[:, halfSize :], mass2)

    def _generate_sorted_range(self, _range):
        ''' Samples a batch of ranges with hands sorted by strength on the board.

        Params:
            range: a NxK tensor in which to store the sampled ranges, where N is 
                the number of ranges to sample and K is the range size'''
        batch_size = _range.size(0)
        self._generate_recursion(_range, arguments.Tensor(batch_size).fill_(1))

    def set_board(self, board):
        ''' Sets the (possibly empty) board cards to sample ranges with.

        The sampled ranges will assign 0 probability to any private hands that
        share any cards with the board.
        
        Params:
            board: a possibly empty vector of board cards'''
        hand_strengths = evaluator.batch_eval(board)    
        possible_hand_indexes = card_tools.get_possible_hand_indexes(board)
        self.possible_hands_count = possible_hand_indexes.sum(0, dtype=torch.uint8).item()
        self.possible_hands_mask = possible_hand_indexes.view(1, -1)
        if not arguments.gpu:
            self.possible_hands_mask = self.possible_hands_mask.bool()
        non_coliding_strengths = arguments.Tensor(self.possible_hands_count)  
        non_coliding_strengths = torch.masked_select(hand_strengths, self.possible_hands_mask)
        _, order = non_coliding_strengths.sort()
        _, self.reverse_order = order.sort() 
        self.reverse_order = self.reverse_order.view(1, -1).long()
        self.reordered_range = arguments.Tensor()
        # self.sorted_range =arguments.Tensor()

    def generate_range(self, _range):
        ''' Samples a batch of random range vectors.
         
        Each vector is sampled indepently by randomly splitting the probability
        mass between the bottom half and the top half of the range, and then
        recursing on the two halfs.

        @{set_board} must be called first.
        
        Params:
            range: a NxK tensor in which to store the sampled ranges, where N is 
                the number of ranges to sample and K is the range size'''
        batch_size = _range.size(0)
        self.sorted_range = arguments.Tensor(batch_size, self.possible_hands_count)
        self._generate_sorted_range(self.sorted_range)
        # we have to reorder the the range back to undo the sort by strength
        index = self.reverse_order.expand_as(self.sorted_range)
        self.reordered_range = self.sorted_range.gather(1, index)
        
        _range.zero_()
        _range[self.possible_hands_mask.expand_as(_range)] = self.reordered_range.view(-1)
