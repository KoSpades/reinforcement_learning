04/16/26

### Learning Objective
- We want to get out from this project: some real coding experience with a project beyond the toy example (Easy21).
- Some more familarity with Deep RL (which we didn't get in Easy21, since we just used tabular lookups and linear function approximators).

We have previous attemped Easy21. But it has a small state space (200 total states) and action space (2 actions). And it worked very well. (just did tabular look ups; the value function approximation also has very small # of features: 36).

In Easy21, the first thing we did was writing a step function with input (state, action), and outputs the next state. Let's start doing that for Gomoku too.

We will try to avoid handcrafted features in this project (having read "The Bitter Lesson"), and try to use methods that leverage searching and learning. 

Maybe we will use a neural net for value function approximations, and self-play, and MCTS for decision-time planning.

### First Problem: State and Action Representation

My initial idea: 

State: i will represent it with 2D, 9 by 9 grid. Each grid is 1 (black), 2 (white), or 0 (unoccupied). so there are total of 3^81 states.

Action: a coordinate. i will scan the board to determine if it's black of white move, then occupy that piece with the right color. so there are 81 possibilities. (max)

Based on Gemini's feedback: I should use a tensor of shape (2, 9, 9), where dimension 0 are the black pieces and plane 1 are the white pieces. And I should represent action with a single integer (0-80). And I should explicitly pass in who is playing to save scanning (which is slow).

I asked more on why a (2, 9, 9) representation is better. Some reasoning comes back as it's easier to check for win conditions (makes sense), and how if later I need a NN it's easier to work with 0's and 1's, so lets' try that.

I asked what kind of libraries I should familiarize myself with, and the suggestions are:
- Pytorch: for any NN related stuff later on
- W&B: for managing statistics and experiments (? sure)
- OpenSpiel: especially its AlphaZero algorithm and documentation. In particular, from the official docs, "AlphaZero is an algorithm for training an agent to play perfect information games from pure self-play. It uses Monte Carlo Tree Search (MCTS) with the prior and value given by a neural network to generate training data for that neural network."

It sounds like exactly what we wanted, so let's use this for our project. Will pick up from here next time by first learning what AlphaZero is all about.

04/17/26

Main pieces to engineer:
- the environment
- search baseline
- learning components
- full self-play system

We will start with the environment, seems like this part requires no knowledge of MCTS or Torch, so we can get started right away :)

### The Environment 

We will code the following for the environment:
- state
- action
- whose turn it is
- step function
- legal move checking
- "who win" (i.e., checking for terminal states)