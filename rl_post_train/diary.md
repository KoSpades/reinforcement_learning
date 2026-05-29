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

Random Idea: this benchmark is potentially one place that we can test the AI factory idea too, independent of the post-training exercise we want to do.

### Fun (Important) things to think about before getting started
1. We need to pick a model: looks like Qwen3-8B is just the right fit.
2. We need to pick a domain to have a sharp focus: looks like airline is a good one.
3. We need a budget: let's aim for $500. This should be sufficient for a portfolio project, and the way we manage cost may also involve some useful thinking later on.

We are ready to get started with the codebase.