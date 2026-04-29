## 04/16/26

## Learning Objective
- We want to get out from this project: some real coding experience with a project beyond the toy example (Easy21).
- Some more familarity with Deep RL (which we didn't get in Easy21, where we just used tabular lookups and linear function approximators).

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

5. Some tensor operations:
- tensor.view(some shape): alter the shape of a tensor without changing its values (e.g. can view a 2D tensor as a 4D one)
- tensor.sum(dim): sum a tensor along the given dimension
- tensor.flatten(): flattens any tensor into a 1D tensor
- tensor.clone(): creates a deep copy of a tensor
- t1 = (t2 condition), where condition is an elementwise comparison: returns a tensor of the same shape with Boolean values (i.e., a Boolean mask)

6. Some torch functions:
- torch.ones(): returns a tensor of 1's with specified shape
- torch.stack([t1, t2, ...]): stack a bunch of tensors with the same shape together
- torch.eye(#): return an identity matrix tensor with specified shape # by # (2D by default)
- torch.fliplr(): flips a 2D tensor left-to-right
- F.conv2D(t1, t2): returns a tensor of convolution scores
- F.conv2D(t1, t2) >= target: return a Boolean tensor 
- F.relu(t1): return a tensor of same shape applying ReLu
- (F.conv2d(player_board, ker) >= 5).any(): return a single boolean tensor. if the previous function finds at least one True

### Checking for win conditions

Here is how we can check for win-conditions, using torch and 2D convolusion:
- initialize a bunch of tensors (5-in-a-col, 5-in-a-row, and the two diagonals)
- sweep through the board and check for where dot product == 5 

A quick note: F.conv2D expects 4D tensors as inputs. Some implications:
- so 5-in-a-row is represented as torch.ones((1, 1, 1, 5), device=device)

### Get a legal move

My thought is to go through the board, find all empty spaces, then randomly pick one.
I looked up the Pytorch way on how to do it, and will use that in the code.

Note: going forward, remember to have state on mps too.

## 04/18/26

Let's get a game going to see it in action, have some pretty print. Then we will move onto some RL stuff.

### Roll out a game with random actions.

This is easy. We solved it with a simple while loop.

### Pretty print.

This seems less important. We generated this part with LLM.

Looks like we have the environment finished, let's read AlphaZero now.

After some research, it looks like we should first try to understand the NN component from AlphaZero, then worry about MCTS later, so we will first study it.
After some further research, we will probably do something like REINFORCE with a NN to start, then later transition into MCTS.

## 04/19/26

We reviewed policy gradient methods and what the REINFORCE algorithm is doing.

## Roadmap Ahead

1. First we do REINFORCE with some kind of NN policy network.
2. Then actor-critic.
3. Then we will do MCTS.

Let's start with REINFORCE.

### First piece of REINFORCE: Getting the policy network

Concretely, a policy network works as follows:
- Is parameterized by some w.
- Take as input the state s.
- Output a probs distribution over all actions.

Our goal is to adjust w to select the good actions.

We need to understand what kind of pieces go into our DNN. After some research, it looks like we should first study what a CNN is. We will watch a lecture on this: https://www.youtube.com/watch?v=f3g1zGdxptI.

Fistly, why the loss function $-log(\pi(a | s, w)) * G$: 
- Minimizing this loss function is equal to saying: For this current (s,a) pair that we are visiting: if G is large and positive for this action a, we want to increase the likelihood of outputting this action. If G is negative, we want to decrease the likelihood of this action.
- So this is exactly what we want our policy to do.

Thing to take away from class:
1. The idea of using a NN is to think of the input as some tensors, the output as some tensors, and build this NN to optimize some loss function.
- For example, for image classification, the input are the raw image pixels, the output can be a distribution of the categories (e.g. cats, dogs, etc.), and the loss function can be if we are right or not (we have the supervised labels.) That is, given image state, predict image class.
- From this perspective, we see how this connects to Gomoku: the input is a tensor of the board state, the output is a distribution of the action, and, as discussed above, we optimize the loss function so we can increasingly take good actions. That is, given board state, predict good actions.
2. We have seen linear layers with an activation. First, what are the "meaning" of linear layers? Second, what are limitations with just linear layers?
- Meaning: A linear layer is a dot product. So a high value means a high match between our data (x), and some kind of "template vector" (weight w).
- Limitation: a full dot product completely ignores the spatial structure of the input data.
3. What are convolution layers?
- They are small 2D structures (i.e., the conv kernels) that sweeps through a 2D image and look for "template matches", which preverses some spatial structure.
- Since they are also just dot products, a typical design to introduce non-linearity is Conv layer -> Activation -> Conv layer -> Activation, etc. etc.
4. More generally, for NN architecture design:
- There are a bunch of operators (like linear, conv, activation) that we can choose from. We want to construct a graph of these operators that makes sense for our task.

This lecture gave us enough info on the intuition of Conv layers and why they may be suited for Gomoku (learning features about the board state).

## 04/20/26

We are doing one more lecture on training CNNs and CNN architectures: https://www.youtube.com/watch?v=aVJy4O5TOk8.


### How to build CNN networks?
1. Other operators to use: LayerNorm, Dropout

2. Where to place activations in a NN?
- Generally placed after linear layers (e.g. fully connected, CNNs)

3. Some popular architectures:
- some 3*3 Conv layers -> pool -> conv -> pool -> FC -> FC (with dim_out equal to the number of categories we have) -> Softmax over all categories
- Resnet

4. How to do weight initializations?
- times it by sqrt(2/dim_in) magically works :D
- why? to prevent the layer mean/std from becoming too large or too small

### How to train CNN networks?
1. Data preprocessing: What transformations do we apply to the data before passing it to the NN?

Subtract per-channel mean and divide by per-channel standard deviation.

This lecture is kinda vague, but the example architecture of CNN network is still helpful. Let's dive back into constructing our policy network.

I think we are ready to create the network and implement REINFORCE.

### High-level TODO List for Implementing any NN
1. Define the input/output tensor

2. Define the model architecture 
- Write an NN.module
- declare layers in init
- define computation in forward
- define the associated optimzer with a learning rate

3. Have your dataset ready.

4. Train
- Feed in input.
- Construct the loss.
- Backpropagate and update.

## 04/21/26

### Some other NN details before we start with the architecture design

1. Should we do pooling layers?
- If we don't need exact spatial output, have a large input, and can afford to lose local details, then we can use pooling.
- Gomoku does not follow any of these three rules: we want the precise board layout to place our stones; input board isn't that large, and don't want to lose any local details. So let's not use pooling at all.

2. What's the benefit of doing padding 1 to conv 3*3 kernels?
- Mostly just so that our board don't shrink too quickly, so we can keep stacking conv layers on top.

3. General questions on how many filters in each conv layer and how many conv layers in total.
- Sounds like 64 filters is the general practice, and for start we can do 5 conv layers.

### Implementing the policy network

Time to write some Torch :) We will do five layers of (Conv->Relu) followed by a FC with (64,9,9 -> 9,9).

This is done! Let's do Reinforce then.

## Implementing REINFORCE

1. How is the per time-step action generated, in generating episodes?

They are generated based on the policy NN. Here is the loop: 1) Observe current state. 2) Feed state to NN to get logits. 3) Do illegal move masking. 4) Turn logits into a probs distribution. 5) Sample a move.

2. I just realized that I am not encoding turn information in the NN. Is it a problem? How do I fix?

Yes, we shall encode turn information. Easy fix is to swap the planes during action generation stage to always have plane[0] correspond to the player who's playing.

3. I also realized: since GOMOKU is a two-player game, after I pick a move, I also need to wait until my opponent picks a move, until I get to the next "state" where I pick an action. Is this true?

Yes. From the lens of self-play, we should then think of the policy network as picking actions for both players. To handle this, we can generate a single game, split it into two trajectories and do our updates. 

4. Some tensor operations:
- t1.unsqueeze(0)/squeeze(0): adds/removes a dimension of size 1 at first position. Typical for adding a "fake batch" dimension for some F to work.
- t1.masked_fill(t2, val): fill the entries where t2 is True with some value
- if not t1.any(): checking if a tensor is all False

5. Some torch functionalities:
- dist = torch.distributions.Categorical(probs=my_probs): gives a probs distribution over a finite set of choices. We can later perform things like action = dist.sample(), and log_prob = dist.log_prob(action) from it.

We are right before actually performing the weight updates using both the black trajectory and white trajectory with the rewards and action log probability, will complete this tomorrow.

## 04/22/26

Let's not get into hyperparameter tuning for now, so let's just use a fixed learning rate.

### How do to weight updates in general in Torch?

1. Define a scalar loss. (for GOMOKU, it's the sum of all log action probs times G)
2. Clear old gradients: optimizer.zero_grad()
3. Back propagation with: loss.backward()
4. Perform weight updates: optimizer.step()

One thing I realized is that the current code has a bug for games that led to a draw, so we need to handle draw games properly.

Fundamentally, since draw games led to zero reward, it will not have an impact on the training, so let's just skip them entirely. 

DONE! We finished implementing REINFORCE. Let's set up something to see it in action/check its effectiveness.

There are definitely many more sound strategies to evaluate its improvement, but let's start with vibe coding a UI so that I can actually play against it, because that will be fun :)

### Saving/load a torch NN
- Save it with: torch.save(my_model.state_dict(), "model_path")
- Load it with: another_model.load_state_dict("model_path", map_location=device)

We actually got to play against it in action! 

### First Set of Results
Trained for 1000 iter (took about 6 minutes).
It now has learned to place 5 in-a-row, which I think is improvement, due to how bad the initial policy is (i.e. random).

But we also found something super interesting: over multiple runs of the game, the NN always seems to be picking the SAME SEQUENCE of moves over and over again. After searching online, it seems like it's a famous problem called the entropy collapse, let's investigate this more. 

## Dealing with entropy collapse

1. Solution 1: add entropy regularization
- We will modify the loss turn by subtracting from it the entire entropy over the trajectory, for some hyperparameter beta. We will use 0.01 to start with as suggested. 
After playing around this parameter (which has a HUGE impact on performance, for the few values we have tried), we think 0.3 may be too large, and we are sticking with 0.01 for now.

2. Solution 2: randomized start board.
- We implemented this. (randomly pick 2 moves in the beginning)

### Potential point of inefficiencies with the current code

It seems like there are many optimizations we can do to make the whole code much more efficient, and we will implmement all of them. But before that, let's actually look at how the loss evolves with with iteration number. 

We added a simple helper that plots the loss by iter number. But the plot is different from a supervised learning plot where the loss monotonically converges to zero, we will come back to make sense of this plot later.

Let's work on some optimizations to the existing code, and see how much efficiency gain we got.
- For now: 1000 iter takes 330 seconds.

1. check_win_cond(): we updated this code to use plain python to only check for the four lines that the lastly placed stone is at. It led to some small speedup, which is fine.

2. Moving training to CPU

This one worked immediately (8X speedup, from 330s -> 45s), since we are not really taking advantage of batches in our code yet, we will make this fix eventually, but for now let's just use GPU.

Tomorrow: let's implement some better baselines: e.g. playing against stable opponents, or something else, that gives us more insights into whether the policy is actually improving. We will do a fully-random baseline, and another baseline where for each move, we check if we can immediately win, if yes do that, else use some freezed NN to output moves.

Another observation: it looks like the network is quite good at making 4 in a row, but not 5. It may be because of the training process, as we are skipping over the final board position (i.e. the winning position), so somehow NN learned that 4 in a row is good, but it did not learn how to finish a game off properly. We should also investigate this.

## 04/23/26

As discussed yesterday, we created two versions of opponents for better evaluation: the first one picks random moves, the second one has the winning/block heuristic, and otherwise picks moves according to a frozen NN.

We started a new opponent.py for this purpose.

### Evaluation against fixed opponents

We will just measure win rate against a chosen set of opponents, out of 100 games, where we randomly alter who starts first. And each game we will do 1 or 2 random moves.

We started a new evaluation.py for this purpose.

We also did some other refactoring. An outline of the repo:
- main.py: the major RL logic there: REINFORCE, generate_episode()
- utils.py: a bunch of generic utility function like step(), check_win_cond(), and get_random_legal_move()
- model.py: contains the NN architecture
- opponent: contains different opponent class with different action heuristics
- evaluation: contains eval related stuff. (e.g., calc_win_rate() between two Player class.)

So mostly a SE day, rather than a RL day :)

### Results:
- 10000 iter beats 1000 iter 99 to 1. This is some evidence that the training is working. 
- 10000 iter lost to FirstOpponent 0 to 100: this is both expected, and strong evidence that we have room for improvement.

Let's look to move into actor-critic hopefully tomorrow.

## 04/25/26

To improve the training pipeline's flexbility, we will implement two additional flags:
- if fresh_train and self-play: we train a NN from scratch, both following the same policy.
- if fresh_train and not self-play: we train a NN from scratch, but pass in a fixed opponent
- if not fresh_train and self-play: we continue training an existing NN, both following the same NN
- if not fresh_train and not self-play: we continue training an existing NN and pass in a fixed opponent

This ends up being a lot of work actually, but we can now train against a fixed opponent now.

### Observation on a fully collapse policy
- We just had a run where we were continuing training from a check point, then then all losses are zero after 10k iters: basically, no training happened.
- After studying this more: it's because the policy has already fully collapsed, so the log probs and entropy both becomes zero, and there are no more losses to be trained.
- We will not try to solve this now, instead, let's really move on from the plain REINFORCE to other algorithms.

### Observation on training REINFORCE directly against FirstOpponent
- Despite throwing about 20K iterations, the loss plot is not changing, just mostly negative. It turns out that the reason is our NN is losing all of its games, so every trajectory/action it took is treated as a negative sign, and REINFORCE is not really introducing any benefits. (i.e, we are not gaining confidence on any winning moves, because we never won.) And this seems like a fundamental problem of the algorithm that throwing in more compute won't help.
- It's time to stop tuning and move on to new algorithms :)

Before doing actor-critic: let's study and understand the meaning of "add a value head".

## 04/27/26

### What are the limitations of plain policy gradient methods? (e.g. REINFORCE)?
- It does not make efficient use of data, when the reward is sparse.
- Suppose we are learning how to talk, and a policy takes 1 step forward and 2 steps back, plain policy gradient will not be able to understand that "one step forward" is making progress, due to reward sparsity.

### Connection between the actor-critic algorithm and the value head
1. The high level idea of "actor-critic" is that there is an Actor: who looks at the current state and chooses an action, and a Critic: who look at the current state or state-action pair and estimates how good it is.
- For actor: We define the policy loss in a way to increase the likelihood of "good actions".
- For critic: We define the value loss in a way to better predict the state values (i.e, predicted value should be close to the actual value.)
3. Current, our NN is outputting a logits over the actions, this is precisely the "Actor portion".
4. We can modify the output layer to also output a scalar value (i.e., the value head), that serves as the Critic for estimating how good the current board state is.
5. Is it a good practice to add the value head in the same NN?
- Yes, and the reason is that the policy and value needs many of the same features about the current board (e.g. where the stones are, open threes/fours, etc.), so having a common NN makes sense and is more efficient.
- This is the same setup as used in AlphaZero.
6. Concept of advantage:
- A(π, s, a) = Q(π, s, a) - V(π, s): how much better is it to take action a at state s than average
- This will be useful in actor-critic.

We have clarified the conceptual part of actor-critic, time to move onto implementation! This should be our next big milestone. To achieve this, our next steps include:

1. Understand how the value head is added, and add it.
2. Write down the actor-critic algorithm by hand.
3. Translate that to code. 

## 04/28/26

### Adding the value head
- Step 1: Move the Conv2d layers out as a common "feature extraction layer".
- Step 2: create two additional NN.sequential() inside the network: one for outputting the policy (with output_dim 9 by 9), and one for outputting the value (with output_dim 1).
- Step 3: To make the value match the range of our reward (1 and -1), we will apply a tanh in the value head.

This is finished.

### The Actor-Critic algorithm on a high level

1. Generate an episode
- sample action from the actor
- store information including the log probs, entropies, and predicted values

2. Compute per-step losses
- actor loss: update the policy using advantage. We will do (G - predicted state value) as the advantage, rather than doing a prediction for the Q(S, A) pair. This is because in games like Gomoku, we really only get a meaningful reward at the end. This is also the AlphaZero style approach.
- critic loss: update the (state) value function using squared error between predicted value (output from the NN) and final reward G. This is intuitive.
- entropy loss: same calculation as before.

3. Sum over the average of all the losses and do gradient updates once for the entire episode.
- Note: this is different from the TD style update where we update after every step. (so our NN can change even during the middle of generating a trajectory.) It looks like updating once after an episode is a sensible practice.

We are ready to code! :) 

## 04/29/26

Finished step 1: updating generate_episode() by tracking the predicted values.

For step 2: Let's first write the code. There seems to be a deeper question here though: since actor loss, critic loss, and regularized loss are summed together, if any one of them is not on the same scale as others, it will dominate/overshadow the others. This seems a bit tricky to handle, and we need a solution.
