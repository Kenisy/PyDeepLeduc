# PyDeepLeduc

Python implementation of [Deepstack-Leduc](https://github.com/lifrordi/DeepStack-Leduc) w/ Numpy, Pytorch 1.4. Works on Windows, Linux.

## Prerequisites

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
2. Create conda environment using environment.yml

## Leduc Hold'em

Leduc Hold'em is a toy poker game sometimes used in academic research (first
introduced in [Bayes' Bluff: Opponent Modeling in Poker](http://poker.cs.ualberta.ca/publications/UAI05.pdf)). 
It is played with a deck of six cards, comprising two suits of three ranks each
(often the king, queen, and jack - in our implementation, the ace, king, and
queen). The game begins with each player being dealt one card privately,
followed by a betting round. Then, another card is dealt faceup as a community
(or board) card, and there is another betting round. Finally, the players
reveal their private cards. If one player's private card is the same rank as
the board card, he or she wins the game; otherwise, the player whose private
card has the higher rank wins.

The game that we implement is No-Limit Leduc Hold'em, meaning that whenever a
player makes a bet, he or she may wager any amount of chips up to a maximum of
that player's remaining stack. There is also no limit on the number of bets and
raises that can be made in each betting round.

## Documentation

Documentation for the original DeepStack Leduc codebase can be found [here](Doc/index.html).
In particular, there is [a tutorial](Doc/manual/tutorial.md) which
introduces the codebase and walks you through several examples, including
running DeepStack.

To run PyDeepLeduc scripts you need to replace `th` with `python` and `.lua` with `.py`

## Differences from DeepStack

- Batch normlization layers were added to make converging faster.
- Training neural network will automatically save the lowest validation loss epoch.
- Starting learning rate will be 1e-2 and decrease to 1e-3 if validation loss doesn't decrease after 10 epochs.

## References

- [Deepstack-Leduc](https://github.com/lifrordi/DeepStack-Leduc)