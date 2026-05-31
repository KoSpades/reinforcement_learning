## 05/27/26

## Learning Objective
- We want to get out from this project: real coding experience with some of the most-to-date RL application, in the context of post training. (PPO, GRPO, etc.) It should be a strong portfolio project for a LLM post-training position at a serious lab.

Easy21 was a nice, first toy project. Gomoku was much more real (REINFORCE, Actor-Critic, MCTS and self-play), and we got much useful experience from it, but it's still a boardgame after all. On the other hand, RL for post-training alignment is its most practical application right now, so we should try it with our own hands.

We should pick a domain with the most real impact too. We need to find this domain/task combination carefully.

After having the domain/task, we will look at which components are important to build, then go from there.

After some exploration, we should start with understanding the tau-3 bench. The way to study it is to first get all the important concepts (i.e., nouns), and really know what those concepts mean, how they are implemented. 

After we have a good understanding of tau-3 bench, we will then plan the major milestones and next steps.

Before we start inspecting the tau-3 bench in details, we will first read the original tau paper by Shunyu.

## 05/28/26

### Understanding the Agent Evaluation Benchmark

1. What is the benchmark evaluating?
- "Can your customer-service agent complete a realistic user request coming from a LLM-simulated user, using backend tools and following specific domain rules, and end up in the correct world/database state?"

2. What do I give to the benchmark as input?
- I need to give an "agent implementation": a program that takes in the current environment, and returns the agent's next action. It will participate in a multi-turn interaction.
- Examples of actions include: a message/a tool call/a final, stop action.

3. What does the benchmark give back to me as output?
- It gives back the evaluation results: for each task, there's a success/failure score; and an aggregate success score across tasks.

4. What does the benchmark contain?
- Domain policies (e.g. for airline domain, it can be something like "cannot rebook after 24h."). Stored as a file.
- Tool definitions (e.g. get_reservation(), search_flights()). Stored as Python functions, with descriptions that explain how to use each function. 
- Task definitions (e.g. a textual description of what the user's goal is, like "change my flight reservation from tonight to tomorrow morning"). Stored as files (JSON, with metadata in addition to task description). Also stores the desired "goal state" that's hidden from the agent - this will be used to evaluate task success/failure.
- User simulator, i.e., an LLM pretending to be the user for the task. Agent doesn't interact with the user simulator directly: the benchmark code sits in between and mediates the interaction.
- Database/world state (e.g. information about reservations, flights, etc.). Stored as structured data.

Random Idea 1: this benchmark is potentially one place that we can test the AI factory idea too, independent of the post-training exercise we want to do.

Random Idea 2: this benchmark is also a good place to work on the LLM for RL (having a seperate policy) idea.

### Fun (Important) things to think about before getting started
1. We need to pick a model: looks like Qwen3-8B is just the right fit.
2. We need to pick a domain to have a sharp focus: looks like airline is a good one.
3. We need a budget: let's aim for $500. This should be sufficient for a portfolio project, and the way we manage cost may also involve some useful thinking later on.

We are ready to get started with the codebase.

## 05/29/26

Bad news: using cloud models, one run of the task one takes a few seconds to complete. But local Qwen3-8b does not complete even after 15 minutes. Something is probably wrong.

It seems like we need to change ollama/Qwen3 to ollama_chat/Qwen3, and that at least got a conversation going.

But, 1 task run in airline took 12 minutes to complete (without succeding). If 1 rollout is taking 12 minutes, this doesn't feel like the right way to get much data for training :) We need to look into better options.

We did some nasty steps to migrate this to Google colab, but at least things are running now. We need an efficient pipeline to do this migration (likely a script?)

But task 1 only took 40 seconds for us to get data. This is massive improvement, so we have to use cloud, and that's the most important next step. Currently testing this with a A100.

Some other TODOs: study the airline domain in close details to understand the databases, the policies, the tools, etc. etc., so that we can diagonose problems and prepare ourselves for the post-training step. The important files are in:
- src/tau2/domains/airline
- data/tau2/domains/airline

Let's also figure out how the default agent is implemented.

Question: but if we collect rollouts from the 50 tasks, then RL post-train on these 50 tasks, aren't we cheating? Let's also figure out the answer to this question.

## 05/30/26

### Commands to get rollouts from Google Colab 

Because running locally to get rollouts is painfully slow, we will do the actual rollouts from Colab. Run the following commands:

```bash
git clone https://github.com/KoSpades/reinforcement_learning.git /content/reinforcement_learning
OPENAI_API_KEY= bash /content/reinforcement_learning/rl_post_train/colab_tau2_airline_qwen.sh
```

Optional arguments after the bash command:
- --num-tasks 10 (how many tasks out of the benchmark do we want to do, default 50)
- --runs 2 (how many runs do we want to do, default 1)

After doing some deep digging, it seems like vLLM is much better than Ollama to use on A100, so we will switch the script to use vLLM to serve Qwen on the cloud. Let's understand why (what ollama and vLLM even are, and why such a big performance discrepancy). It seems like with vLLM, we can also get better max-concurrency (Ollama seems to only do 1). 

Let's also study what's a good max-concurrency to use, before we later do serious experiments (which will take a long time). 
- we experimented with max_concurrency=16 vs =8. The speed up was little from 16 to 8 (12 minutes down to 10), but results at 8 has less agent timeout. With this in consideration, we will fix on the max_concurrecy of 8.

vLLM seem to give the maxContextLength exceeded error quite a few times, we should also look into why.
- After checking some math, it seems like we can 4x the context length to 32K with little trouble. So let's do that.
- Doing this seems to solve the context length problem entirely.

Currently, we are giving a 1 min max generation time for the agent. We may also need to adjust this based on how many timeouts we are actually getting. 
- We decided to settle down on 2.

