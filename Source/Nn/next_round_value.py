''' Uses the neural net to estimate value at the end of the first betting round.
@classmod next_round_value'''

from Source.Settings.arguments import arguments
from Source.Settings.constants import constants
from Source.Settings.game_settings import game_settings
from Source.Game.card_tools import card_tools
from Source.Nn.bucketer import Bucketer
import torch

class NextRoundValue:
    def __init__(self, nn):
        ''' Constructor.

        Creates a tensor that can translate hand ranges to bucket ranges
        on any board.
        @param nn the neural network'''
        super().__init__()
        self._values_are_prepared = False
        self.nn = nn
        self._init_bucketing()

    def _init_bucketing(self):
        ''' Initializes the tensor that translates hand ranges to bucket ranges.
        @local'''
        self.bucketer = Bucketer()
        self.bucket_count = self.bucketer.get_bucket_count()
        boards = card_tools.get_second_round_boards()
        self.board_count = boards.size(0)
        self._range_matrix = arguments.Tensor(game_settings.card_count, self.board_count * self.bucket_count ).zero_()
        self._range_matrix_board_view = self._range_matrix.view(game_settings.card_count, self.board_count, self.bucket_count)

        for idx in range(self.board_count):
            board = boards[idx]
            
            buckets = self.bucketer.compute_buckets(board)
            class_ids = torch.arange(0, self.bucket_count)

            if arguments.gpu:
                buckets = buckets.cuda() 
                class_ids = class_ids.cuda()
            else:
                class_ids = class_ids.float()

            class_ids = class_ids.view(1, self.bucket_count).expand(game_settings.card_count, self.bucket_count)
            card_buckets = buckets.view(game_settings.card_count, 1).expand(game_settings.card_count, self.bucket_count)

            # finding all strength classes      
            # matrix for transformation from card ranges to strength class ranges 
            self._range_matrix_board_view[:, idx, :][torch.eq(class_ids, card_buckets)] = 1

        # matrix for transformation from class values to card values
        self._reverse_value_matrix = self._range_matrix.T.clone()
        # we need to div the matrix by the sum of possible boards (from point of view of each hand)
        weight_constant = 1/(self.board_count - 2) # count
        self._reverse_value_matrix.mul_(weight_constant)

    def _card_range_to_bucket_range(self, card_range, bucket_range):
        ''' Converts a range vector over private hands to a range vector over buckets.
        @param card_range a probability vector over private hands
        @param bucket_range a vector in which to store the output probabilities
        over buckets
        @local'''
        torch.mm(card_range, self._range_matrix, out=bucket_range)

    def _bucket_value_to_card_value(self, bucket_value, card_value):
        ''' Converts a value vector over buckets to a value vector over private hands.
        @param bucket_value a value vector over buckets
        @param card_value a vector in which to store the output values over 
        private hands
        @local'''
        torch.mm(bucket_value, self._reverse_value_matrix, out=card_value)

    def _bucket_value_to_card_value_on_board(self, board, bucket_value, card_value):
        ''' Converts a value vector over buckets to a value vector over private hands
        given a particular set of board cards.
        @param board a non-empty vector of board cards
        @param bucket_value a value vector over buckets
        @param card_value a vector in which to store the output values over 
        private hands
        @local'''
        board_idx = card_tools.get_board_index(board)
        board_matrix = self._range_matrix_board_view[:, board_idx, :].T.clone()
        serialized_card_value = card_value.view(-1, game_settings.card_count)
        serialized_bucket_value = bucket_value[:, :, board_idx, :].clone().view(-1, self.bucket_count)
        torch.mm(serialized_bucket_value, board_matrix, out=serialized_card_value)

    def start_computation(self, pot_sizes):
        ''' Initializes the value calculator with the pot size of each state that
        we are going to evaluate.

        During continual re-solving, there is one pot size for each initial state
        of the second betting round (before board cards are dealt).
        @param pot_sizes a vector of pot sizes
        betting round ends'''
        self.iter = 0
        self.pot_sizes = pot_sizes.view(-1, 1).clone()
        self.batch_size = pot_sizes.size(0)

    def get_value(self, ranges, values):
        ''' Gives the predicted counterfactual values at each evaluated state, given
        input ranges.

        @{start_computation} must be called first. Each state to be evaluated must
        be given in the same order that pot sizes were given for that function.
        Keeps track of iterations internally, so should be called exactly once for
        every iteration of continual re-solving.
        # 
        @param ranges An Nx2xK tensor, where N is the number of states evaluated
        (must match input to @{start_computation}), 2 is the number of players, and
        K is the number of private hands. Contains N sets of 2 range vectors.
        @param values an Nx2xK tensor in which to store the N sets of 2 value vectors
        which are output'''
        assert ranges != None and values != None
        assert(ranges.size(0) == self.batch_size)
        # TODO recheck
        self.iter = self.iter + 1
        if self.iter == 1:
            # initializing data structures
            self.next_round_inputs = arguments.Tensor(self.batch_size,  self.board_count, (self.bucket_count * constants.players_count + 1)).zero_()
            self.next_round_values = arguments.Tensor(self.batch_size, self.board_count, constants.players_count,  self.bucket_count ).zero_()
            self.transposed_next_round_values = arguments.Tensor(self.batch_size, constants.players_count, self.board_count, self.bucket_count)
            self.next_round_extended_range = arguments.Tensor(self.batch_size, constants.players_count, self.board_count * self.bucket_count ).zero_()
            self.next_round_serialized_range = self.next_round_extended_range.view(-1, self.bucket_count)
            self.range_normalization = arguments.Tensor()
            self.value_normalization = arguments.Tensor(self.batch_size, constants.players_count, self.board_count)
            # handling pot feature for the nn
            nn_bet_input = self.pot_sizes.clone().mul(1/ arguments.stack)
            nn_bet_input = nn_bet_input.view(-1, 1).expand(self.batch_size, self.board_count)
            self.next_round_inputs[:, :, -1].copy_(nn_bet_input)
        
        # we need to find if we need remember something in this iteration
        use_memory = self.iter > arguments.cfr_skip_iters
        if use_memory and self.iter == arguments.cfr_skip_iters + 1:
            # first iter that we need to remember something - we need to init data structures
            self.range_normalization_memory = arguments.Tensor(self.batch_size * self.board_count * constants.players_count, 1).zero_()
            self.counterfactual_value_memory = arguments.Tensor(self.batch_size, constants.players_count, self.board_count, self.bucket_count).zero_()

        # computing bucket range in next street for both players at once
        self._card_range_to_bucket_range(ranges.view(self.batch_size * constants.players_count, -1), self.next_round_extended_range.view(self.batch_size * constants.players_count, -1))
        # TODO recheck
        self.range_normalization = self.next_round_serialized_range.sum(dim=1, keepdim=True)
        rn_view = self.range_normalization.view(self.batch_size, constants.players_count, self.board_count)
        for player in range(constants.players_count):
            self.value_normalization[:, player, :].copy_(rn_view[:, 1 - player, :])
        if use_memory:
            self.range_normalization_memory.add_(self.value_normalization.view(self.range_normalization_memory.shape))
        # eliminating division by zero
        self.range_normalization[torch.eq(self.range_normalization, 0)] = 1
        self.next_round_serialized_range.div_(self.range_normalization.expand_as(self.next_round_serialized_range))
        serialized_range_by_player = self.next_round_serialized_range.view(self.batch_size, constants.players_count, self.board_count, self.bucket_count)
        for player in range(constants.players_count):
            # TODO recheck
            self.next_round_inputs[:, :, player * self.bucket_count : (player + 1) * self.bucket_count].copy_(self.next_round_extended_range[:,player, :].view(self.next_round_inputs[:, :, player * self.bucket_count : (player + 1) * self.bucket_count].shape))

        # usning nn to compute values 
        serialized_inputs_view= self.next_round_inputs.view(self.batch_size * self.board_count, -1)
        serialized_values_view= self.next_round_values.view(self.batch_size * self.board_count, -1)

        # computing value in the next round
        self.nn.get_value(serialized_inputs_view, serialized_values_view)

        # normalizing values back according to the orginal range sum
        normalization_view = self.value_normalization.view(self.batch_size, constants.players_count, self.board_count, 1).transpose(1,2)
        self.next_round_values.mul_(normalization_view.expand_as(self.next_round_values))
            
        self.transposed_next_round_values.copy_(self.next_round_values.transpose(2,1))
        # remembering the values for the next round
        if use_memory:
            self.counterfactual_value_memory.add_(self.transposed_next_round_values)
        # translating bucket values back to the card values
        self._bucket_value_to_card_value(self.transposed_next_round_values.view(self.batch_size * constants.players_count, -1), values.view(self.batch_size * constants.players_count, -1)) 

    def get_value_on_board(self, board, values):
        ''' Gives the average counterfactual values on the given board across previous
        calls to @{get_value}.

        Used to update opponent counterfactual values during re-solving after board
        cards are dealt.
        @param board a non-empty vector of board cards
        @param values a tensor in which to store the values'''
        # check if we have evaluated correct number of iterations
        assert(self.iter == arguments.cfr_iters )
        batch_size = values.size(0)
        assert(batch_size == self.batch_size)

        self._prepare_next_round_values()

        self._bucket_value_to_card_value_on_board(board, self.counterfactual_value_memory, values)

    def _prepare_next_round_values(self):
        ''' Normalizes the counterfactual values remembered between @{get_value} calls
        so that they are an average rather than a sum.
        @local'''
        assert(self.iter == arguments.cfr_iters )

        # do nothing if already prepared
        if self._values_are_prepared:
            return

        # eliminating division by zero
        self.range_normalization_memory[torch.eq(self.range_normalization_memory, 0)] = 1
        serialized_memory_view = self.counterfactual_value_memory.view(-1, self.bucket_count) 
        serialized_memory_view.div_(self.range_normalization_memory.expand_as(serialized_memory_view))

        self._values_are_prepared = True



