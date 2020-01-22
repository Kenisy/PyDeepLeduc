import torch
class params:
    # whether to run on GPU
    gpu = False
    device = torch.device('cpu')
    # list of pot-scaled bet sizes to use in tree
    bet_sizing = [1]
    # the number of betting rounds in the game
    streets_count = 2
    # the size of the game's ante, in chips
    ante = 100
    # the size of each player's stack, in chips
    stack = 1200
    # the number of iterations that DeepStack runs CFR for
    cfr_iters = 1000
    # the number of preliminary CFR iterations which DeepStack doesn't factor into the average strategy (included in cfr_iters)
    cfr_skip_iters = 500
    # how many poker situations are solved simultaneously during data generation
    gen_batch_size = 10
    # how many poker situations are used in each neural net training batch
    train_batch_size = 100
    # path to the solved poker situation data used to train the neural net
    data_path = '../Data/TrainSamples/'
    # path to the neural net model
    model_path = '../Data/Models/'
    # the neural net architecture
    net = [500, 500, 500]
    # how many epochs to train for
    epoch_count = 10
    # how many solved poker situations are generated for use as training examples
    train_data_count = 100
    # how many solved poker situations are generated for use as validation examples
    valid_data_count = 100
    # learning rate for neural net training
    learning_rate = 1e-3

arguments = params()
assert(arguments.cfr_iters > arguments.cfr_skip_iters)
if arguments.gpu and torch.cuda.is_available():
    arguments.device = torch.device('cuda')