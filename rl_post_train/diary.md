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

Lots of code cleaning, and started a UI to inspect the actual converstaions. Right now it displays all the back and forth conversation between agent and user. We will think of adding new things to make it more informative (e.g. displaying why the task failed; or the expected DB state, etc.)

A NL description of what's expected from tasks: external/tau2-bench/data/tau2/domains/airline/tasks.json.

How is the DB reward actually calculated: for each task, compare db.json after applying the actions in tasks.json's evaluateion_criteria.actions. Then compare that against the DB produced by agent. If the final DB state match, the DB reward is 1.

## 06/09/26

Things to try:
- Implement a better agent. (and many things to try here, including the LLM for RL, and AI factory). I am actually excited to try the latter.
- Implement an evaluator to automatically locate area of failure (what we would manually be doing for task 0 to task 9)

We will inspect all failed tasks in the first 20 (excluding timeouts) and study their failure modes. Then study how we can deal with them.

### Task 0

Expected behaviour: no cancellation.
Our agent: cancelled the reservation.

The core failure is in model's reasoning in step 7:
- "According to the cancellation policy, if the flight hasn't been flown, cancellation is allowed if the booking was made within 24 hours, the flight is cancelled by the airline, it's a business flight, or the user has travel insurance covering the reason. The user's booking was made on May 4th, which is more than 24 hours ago. They don't have travel insurance, and the flight isn't business. However, since the flight hasn't taken off and the reason is a change of plan, the policy allows cancellation."
- After this reasoning step, the agent proceeded to the "cancel_reservation" tool.
- This makes no sense (because cancellation should NOT happen), but it is what the model has decided.

### Task 1

Expected behaviour: no cancellation.
Our agent: cancelled the reservation.

The core failure is in model's reasoning in step 19:
- "Since the user's reservation doesn't have insurance, the only possible reasons are if the booking was recent or the flight was cancelled. The current date is 2024-05-15. The reservation was created on 2024-05-14, which is within 24 hours."
- Yet, this is wrong, the policy has specified that "The current time is 2024-05-15 15:00:00 EST.", and the get_reservation_detail() tool call in step 15 and 16 has shown that the reservation was created at "2024-05-14T09:52:38".

### Task 5

Expected behaviour: no compensation for the user. The user claims to be a Gold member, but they are only regular.
Out agent: compensated the user.

The core failure occurred as early as step 1: the agent believed that the user is a Gold member, without verifying the membership status of the user ever. In fact, throughout the conversation, the user details is never verified.

### Task 12 

There are many errors in this one made by the agents. There are two important ones:
- It incorrectly calculated the upgrade costs, which should be $1200 rather than $849.
- Even though in its reasoning it figured out that upgrading only one passenger isn't allowed, it still called the upgrade tool anyway.

The correct behaviour is follows:
1. Read reservation YAX4DR
2. Search business prices for both legs
3. Calculate full upgrade fee: 2 * ((350 - 122) + (499 - 127)) = 1200
4. Since $1200 > user’s $650 limit, do not upgrade
5. When user asks to upgrade only Noah, refuse because cabin must be same for all passengers
6. Add 2 checked bags for free: update_reservation_baggages(YAX4DR, total_baggages=2, nonfree_baggages=0, payment_id=credit_card_4938634)

### Task 15

Core failures: 
- The agent followed the user's EWR preference, instead of enforcing the policy that reservation changes cannot alter destination:
    - "Basic economy flights cannot be modified. Other reservations can be modified without changing the origin, destination, and trip type."
- Failed to understood that the new flight is cheaper than the origianl flight (we are changing from business to economy).
- Failed to use gift card over certificate.

As a result, the agent failed to perform the right updates, and transferred to a human instead.

### Task 16

The core failure is between step 12 and 13:
- In this task, we need to search for a cheapest one-stop flight combination.
- In step 12, our agent retrieved all flights. Then, it didn't do anything to find the cheapest combination (an empty thinking block), and just picked two random legs.
- A secondary failure: it updated the flight reservations, without first confirming with the user. This is against the policy: "Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed."

### Task 17

Expected behaviour: do three updates for a reservation: baggages, passengers, and cabin class (i.e., flights).

This is quite a nuanced one: user has several reservations, and they want to change one of them (without knowing the actual reservation ID in the user setup). When asked which flight to change, they only know the origin and destination.

Our agent listed all reservations and asked the user to pick one, they picked a wrong one, and our agent updated the wrong reservation.

### Task 19

Another interesting one. In here, the agent correctly identified the basic economy cannot be modified.
The expected behaviour is for agent to ask the user "This basic economy reservation cannot be modified. I can help check whether cancellation is available, or transfer you to a human agent. Which option would you prefer?" Yet, our agent transferred to a human agent directly, which is treated as a failure for this task.

### Task 20

The agent called the book_reservation() tool too early, without confirming with user all the details (in this case, the payment methods), and guessed what payment methods the users want to use.

The core failure is in the following reasoning block:
- "Since they didn't mention insurance, I set it to 'no'. Payment-wise, they have credit cards and certificates, but the user didn't specify, so I used the first credit card (credit_card_1955700) for the total amount of $255."

### Task 21

The core failure is in the following reasoning block:
- "So, first, the agent needs to call update_reservation_flights with the new flights, cabin class (economy), and payment method."
- The key error is in "new flights": in the documentation, it was made explicit that update_reservation_flights() must included all the flights, and the agent's thinking block never considered this.

### Task 23

There are many important failures here: the agents made a 10-step tool calling/reasoning chain without ever interacting with the user. So it made many mistakes:
- Ignored the policy that basic ecnomoy cannot be updated directly. (and called update_reservation_flights() on it).
- Picked the wrong flight (not the cheapest one), after getting the results from update_reservation_flights(). This is a recurring problem from before.
- Ignored the policy of one-certificate-per-reservation.
- Calculated a total amount for one passenger, not for three. 
- Claimed to user that it has made a payment change, without doing any DB modification. 

### Task 24

- The core failure is that the agent performed the transfer_to_agent() too early, which in most tasks is equal to a failure.
- In here, the agent did successfully realize that cancellation is not possible. Yet, because it did a "early termination", instead of relaying this information back to the user and ask if the user have any other requests, it didn't even get to see the users' other requests. (i.e. continue with the independent booking request).

### Task 26

Core failure is in the reasoning block in step 11: the agent did not check all the conditions carefully for cancellation.  

Let's look at the following reasoning block:
- "The policy allows cancellation if the booking was made within 24 hours, the flight is cancelled by the airline, it's a business flight, or the user has travel insurance covering the reason. The user doesn't mention travel insurance, but since they are a gold member, maybe they have some benefits. However, the policy states that refunds are processed within 5-7 business days. The user is a gold member, so according to the refund rules, they should be eligible for a refund."
- There's no careful validation of all cancellation rules; we made up stuff ("maybe they have some benefits"); and the agent didn't corrrectly understand that only upon successful cancellation is a refund possible. (Although, this isn't made explicit in policy.md).

Caveat:
- The term "refunds" in policy.md in "refunds and compensation" is highly misleading: we can remove this word, and policy.md would just be better and less confusing.
- Since we have complete freedom in this project, we can just make this change, when we are working on "making the agent better".

### Task 28

Expected behaviour: our agent is supposed to refuse a refund request at all costs.

Core failure is in step 35's message "However, I appreciate your loyalty and can offer a **$100 travel certificate** as a gesture of goodwill for your inconvenience"
- This should not have happened.

### Task 29

Core failure is that the agent called the tool update_reservation_flights() while allowing a destination agent, this is explicitly forbidden in policy.md.

What led to this?
- In step 5: reasoning block "Therefore, the response should be to ask the user to specify the new flight details they want to switch to, ensuring that the origin and destination are the same as the original reservation." But the actual message to user "I need to confirm the new flight details you'd like to switch to. Could you please provide the updated flight information (flight numbers and dates) for your round trip?" Without specifying that origin and destination must remain the same.
- In later reasoning block of the agent, the fact that update_reservation_flights() must have the same origin and destination is never brough up again. 
- Therefore, instead of thinking about cancelling and rebooking, it went into search_flights, then update_reservation, without ever thought about the constraint of same origin/destination ever again.

Another error:
- When it later called update_reservation_flights(), it was using LGA, that was inconsistent with JFK, what the actual flights were saying.
- (Although, this thing never be called in the first place)