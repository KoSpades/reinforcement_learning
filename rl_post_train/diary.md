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
- we experimented with max_concurrency=16 vs =8. The speed up was little from 16 to 8 (12 minutes down to 10), but results at 8 has less agent timeout. With this in consideration, we will fix on the max_concurrecy of 8. This is a 50X speedup for collecting rollouts vs. yesterday :)

vLLM seem to give the maxContextLength exceeded error quite a few times, we should also look into why.
- After checking some math, it seems like we can 4x the context length to 32K with little trouble. So let's do that.
- Doing this seems to solve the context length problem entirely.

Currently, we are giving a 1 min max generation time for the agent. We may also need to adjust this based on how many timeouts we are actually getting. 
- We decided to settle down on 2 min.

### Project Settings
- model: Qwen3-8B
- serving: vLLM
- machine: Colab A100 80GB
- domain: airline
- max_model_len: 32768
- max_concurrency: 8
- agent_response_timeout: 120s

## 05/31/26

We have a reasonably efficient pipeline on the cloud to gather rollouts.Let's switch gear a bit today and do some deep dive into the airline domain.

## The Airline Domain Data Model

There are three major tables: 
- Users: stores customer accounts (who the users are)
- Reservations: bookings of trips made by customers (what users have booked)
- Flights: scheduled flights offered by the airline (what the airline provides)

**User**:
- user_id: primary key
- name
- address
- email
- dob
- payment_methods: [CreditCard, GiftCard, Certificate]
    - **CreditCard**: {"credit_card", brand: str, last_four}
    - **GiftCard**: {"gift_card", amount, id}
    - **Certification**: {"certificate", amount}
- saved_passengers: a user can save multiple passenger profiles in their account for later bookings
    - List[**Passenger**]: {first_name, last_name, dob}
- membership: ["gold"; "silver"; "regular"]
- reservations: a list of foreign key references to **Reservation**'s reservation_id

**Reservation**:
- reservation_id: primary key
- user_id: foreign key to user_id in **User**
    - who made the reservation
- origin: str; "ORD"
- destination
- flight_type: ["round_trip"; "one_way"]
- cabin: ["business", "economy", "business_economy"]
- flights: a list of flights in the reservation
    - List[**ReservationFlight**]: {flight_number, origin, destination, date, price}
- payment_history: a list of payments
    - List[**Payment**]: {payment_id, amount}
- passengers: a list of passengers on the reservation
    - List[**Passenger**]: {first_name, last_name, dob}
- created_at: str, timestamp when trip was created
- total_baggages: int
- nonfree_baggages: int, number of paid bags
- insurance: ["yes", "no"]
- status: [none, "cancelled"]

**Flight**
- flight_number: primary key
- origin
- destination
- scheduled_departure_time_est: str
- schedules_arrival_time_est: str
- dates: a dict where keys are dates ("2015-08-02"), and values are **FlightDateStatus**
    - **Available**: {status: "available", available_seats: dict[CabinClass, int], prices: dict[CabinClass, int]}
    - **Cancelled**: {"cancelled"}
    - **Delayed**: {"delayed", estimated_departure_time, 
    estimated_arrival_time}
    - **OnTime**: {"on_time", estimated_departure_time, estimated_arrival_time}
    - **Flying**: {"flying", actual_departure_time, estimated_arrival_time}
    - **Landed**: {"landed", actual_departure_time_est, actual_arrival_time_est}

We studied the data models closely, and the three core DBs (flights, users, reservations). We will do tools next, then start with individual tasks one by one to understand what's the expected outcome and their failure modes, then study how the default agent is implemented.

## 06/04/21

A first inspection shows that there are 14 tools in total, this is actually a small enough action space that can potentially be brute forced even with a value-table lookup -> this may be a useful insight later on.

## The 14 Airline Domain Tools

Read Operations

1. get_reservation_details(reservation_id):
    - output: info about a Reservation (all its fields)
    - impl: direct call to Reservation DB
2. get_user_details(user_id):
    - output: info about a User
    - impl: direct call to User DB
3. list_all_airports():
    - output: a hardcoded list of airports with their codes
4. search_direct_flight(origin, dest, date):
    - input: "ORD", "LGA", "2014-01-01"
    - output: all DirectFlgihts between origin and dest on date
    - impl: fetch from DB a list of DirectFlight meeting the requirements. {flight_no, origin, dest, arrival/departure_time, seats, price, etc.}
5. search_one_stop_flight(origin, dest, date):
    - output: all pairs of DirectFlights between origin and dest
    - impl: outer loops fetch all departing flights from origin, inner loops fetches all arriving flights at dest.
6. get_flight_status(flight_no, date):
    - output: status like "cancelled", "on_time", etc.
    - impl: direct call to Flight DB
    - note: (flight_no, date) uniquely identifies a flight instance

Write Operations

7. book_reservation(user_id, origin, dest, flight_type, cabin, flights, passengers, payment_methods, total_bags, nonfree_bgas, insurance):
    - input: these are essentially the fields to create a Reservation
    - impl: 
        - Look up User.
        - Check all Flights and their seats' availability.
        - Calculate total fee; verify payment methods are valid; and ensure these two amounts match exactly.
        - If all valid, deduct GiftCard balance or remove used Certificate; reduce available seats on booked flights; adds Reservation to DB; associate the Reservation with the right User.
    - note: all above steps can lead to their associated failure modes.

8. cancel_reservation(reservation_id):
    - intput: Reservation ID to cancel
    - output: 1) status of reservation is set to "cancelled". 2) negative (refunded) amounts are appended to Reservation.payment_history
    - note: 1) seats are not actually released from the flights, 2) no refunding actually happens.

9. send_certificate(user_id, amount):
    - impl: picks from 3 hardcoded certificate IDs, then assign the first free one to User.payment_methods. Else, raise an error.

10. update_reservation_baggages(reservation_id, total_bags, nonfree_bags, payment_id):
    - impl: based on the new bagagge info, check if payment_id can successfully accomodate teh change. If yes, update the reservation.
    - note: 
        - if payment_method is not yet in Reservation, add it to Reservation.payment_history.
        - failure modes: you cannot use Certificates to update; payment_method not found for user, etc.

11. update_reservation_passengers(reservation_id, passengers: List[Passenger]):
    - impl: update a Reservation's passengers field
    - note: number of passengers must match up exactly.

12. update_reservation_flights(reservation_id, payment_id, cabin, flights):
    - impl:
        - Given user's requirements for the flight instances (flight number + date) and cabin, first calculate the new amount needs to be paid. 
        - Then make a new Reservation.payment_history if payment is successful.
        - Update the reservation instance.
    - note: failure modes: flights not available; payment failure; etc.

Generic Operations

13. calculate(expression):
    - impl: calculate the result of a math expression.

14. transfer_to_human_agent(summary):
    - impl: returns a literal string "Transfer successful!"

These are all the 14 tools in the airline domain. The next step is to understand the default agent's implementation. 

## 06/06/26

Looks like the relevant agent logic is in the following places. We will study them:
- agent/llm_agent.py:24-135
- utils/llm_utils.py:355-469
- orchestrator/orchestrator.py:932-988

## Default Agent logic

Orchestrator:

During the agent's turn, it either sends the simulated user a message or makes a tool call:
- Sending a message: adding it to conversation. User then gets control.
- Making a tool call: agent sees the result, and immediately gets control again.

Agent: 

In summary, the agent just repeated called generate() in each turn, where generate() receives the following inputsL
- system_prompt: 
    - You are a customer service agent that helps the user according to the \<policy> provided below.
    - In each turn you can either send a message to the user, or make a tool call. You cannot do both at the same time.
    - Try to be helpful and always follow the policy. Already make sure you generate valid JSON only.
    - the actual \<policy> text: for each domain, the entire policy is passed into every conversation. For airline domain, this is a 170-line file.
- messages: every piece of UserMessage, ToolMessage, and Assistant(Agent)Message, generated during the entire conversation history.
- all tool definitions
- model: which model does generate() use

Then, the model picks between 1) doing a tool call, or 2) returns a message. If it pickes a specific tool, this information is available in the response object.

### How does the model know how to pick?
- The underlying transformer architecture doesn't change. 
- Picking a tool => the model outputs a structured output corresponding to a tool call, e.g. a JSON.
- Conceptually, it's similar to in-context programming. 
    - The pretrained model learned the general behaviour of "when the prompt contains tool definitions, I need to decide between return a regualar NL message, or a structured output matching some tool definitions in the prompt"
- The model needs no special knowledge about the domain-specific tools (e.g. airline). It just needs the tool definitions as context.

At its core, the default agent is very dumb: paste all policies, paste all conversation history (including tool results), then rely on the underlying LLM for picking tools.

## 06/06/26

Lots of code cleaning, and started a UI to inspect the actual converstaions. Right now it displays all the back and forth conversation between agent and user. We will think of adding new things to make it more information (e.g. displaying why the task failed; or the expected DB state, etc.)