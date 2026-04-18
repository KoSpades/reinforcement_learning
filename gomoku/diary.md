## 04/16/26

## Learning Objective
- We want to get out from this project: some real coding experience with a project beyond the toy example (Easy21).
- Some more familarity with Deep RL (which we didn't get in Easy21, since we just used tabular lookups and linear function approximators).

We have previous attemped Easy21. But it has a small state space (200 total states) and action space (2 actions). And it worked very well. (just did tabular look ups; the value function approximation also has very small # of features: 36).

In Easy21, the first thing we did was writing a step function with input (state, action), and outputs the next state. Let's start doing that for Gomoku too.

We will try to avoid handcrafted features in this project (having read "The Bitter Lesson"), and try to use methods that leverage searching and learning. 

Maybe we will use a neural net for value function approximations, and self-play, and MCTS for decision-time planning.

## First Question: State and Action Representation

My initial idea: 

State: i will represent it with 2D, 9 by 9 grid. Each grid is 1 (black), 2 (white), or 0 (unoccupied). so there are total of 3^81 states.

Action: a coordinate. i will scan the board to determine if it's black of white move, then occupy that piece with the right color. so there are 81 possibilities. (max)

Based on some search: I should use a tensor of shape (2, 9, 9), where dimension 0 are the black pieces and plane 1 are the white pieces. And I should represent action with a single integer (0-80). And I should explicitly pass in who is playing to save scanning (which is slow).

I asked more on why a (2, 9, 9) representation is better. Some reasoning comes back as it's easier to check for win conditions (makes sense), and how if later I need a NN it's easier to work with 0's and 1's, so lets' try that.

I asked what kind of libraries I should familiarize myself with, and the suggestions are:
- Pytorch: for any NN related stuff later on
- W&B: for managing statistics and experiments (? sure)
- OpenSpiel: especially its AlphaZero algorithm and documentation. In particular, from the official docs, "AlphaZero is an algorithm for training an agent to play perfect information games from pure self-play. It uses Monte Carlo Tree Search (MCTS) with the prior and value given by a neural network to generate training data for that neural network."

It sounds like exactly what we wanted, so let's use this for our project. Will pick up from here next time by first learning what AlphaZero is all about.

## 04/17/26

Main pieces to engineer:
- the environment
- search baseline
- learning components
- full self-play system

We will start with the environment, seems like this part requires no knowledge of MCTS or Torch, so we can get started right away :)

## The Environment 

We will code the following for the environment:
- state
- action
- whose turn it is
- step function
- legal move checking
- "who win" (i.e., checking for terminal states)

## Second question: On PyTorch

I can do a for-loop for checking for win-conditions, but that will likely be slow for later operations. So I looked up good practices here.
Based on my research, it's better to use PyTorch for these workloads.

### Let's use 2D convolution

2D convolution is a process where a small grid (the kernel) with user-defined shape slides over a larger grid (the board), and calculates a dot product at every position. 

In gomoku: checking if any resulting dot product matches our target (5) helps checking winning conditions.

### Some Pytorch basics:

1. What does this do? "device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")" and "x_gpu = x.to(device)"

It moves x to GPU, making parallel computations faster and increases throughput.

2. What is "import torch.nn.functional as F"?

"functional" contains functions for common NN operations, such as activation/loss function, pooling, etc. For our purpose, it has something called "F.conv2d".

3. If I perform a F.conv2d between one tensor on CPU, another on MPS(GPU), what happens?

It will throw an error, because PyTorch operations require all tensors to be on the same device.

4. It seems like the general practice with Torch is to send anything "number_like" to GPU, we will remember this going forward.

5. Some basic syntax:
- torch.ones(): returns a tensor of 1's with specified shape
- torch.stack([t1, t2, ...]): stack a bunch of tensors with the same shape together
- torch.eye(#): return an identity matrix tensor with specified shape # by # (2D by default)
- tensor.view(some shape): alter the shape of a tensor without changing its values (e.g. can view a 2D tensor as a 4D one)
- torch.fliplr(): flips a 2D tensor left-to-right
- F.conv2D(t1, t2): returns a tensor of convolution scores
- F.conv2D(t1, t2) >= target: return a Boolean tensor 
- (F.conv2d(player_board, ker) >= 5).any(): return a single boolean tensor. if the previous function finds at least one True

### Checking for win conditions

Here is how we can check for win-conditions, using torch and 2D convolusion:
- initialize a bunch of tensors (5-in-a-col, 5-in-a-row, and the two diagonals)
- sweep through the board and check for where dot product == 5 

A quick note: F.conv2D expects 4D tensors as inputs. Some implications:
- so 5-in-a-row is represented as torch.ones((1, 1, 1, 5), device=device)

Note: going forward, remember to have state on mps too.

## 04/17/26

Let's get a game going to see it in action, have some pretty print. Then we will move onto some RL stuff.
