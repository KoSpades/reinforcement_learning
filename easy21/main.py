import random
import numpy as np
import matplotlib.pyplot as plt


def generate_initial_state():
     player_first = random.randint(1, 10)
     dealer_first = random.randint(1, 10)
     return (player_first, dealer_first)


def generate_new_card():
    card_num = random.randint(1, 10)
    card_color = random.randint(1, 3)
    if card_color < 2:
        return -card_num
    return card_num


def step(state, action):
    """
    INPUT:
    state: dealer's first card, player's sum
    action: 1 ("hit") or 0 ("stick")
    OUTPUT:
    a sample of the next state, and a reward.
    If player sticks, step runs the game to a terminal state.
    """
    player_sum, dealer_sum = state
    if action:
        new_card = generate_new_card()
        player_sum += new_card
        if player_sum > 21 or player_sum < 1:
            # player busted
            # print("Player busted")
            # print(player_sum, dealer_sum)
            return (None, None), -1
        else:
            # player has not busted: game continues
            return (player_sum, dealer_sum), 0
    else:
        # print("roll out the game")
        # In this cases we roll out the dealer's card until termination
        while dealer_sum < 17:
            new_card = generate_new_card()
            dealer_sum += new_card
            # print(player_sum, dealer_sum)
            if dealer_sum > 21 or dealer_sum < 1:
                 # dealer busted
                 # print("Dealer busted")
                 # print(player_sum, dealer_sum)
                 return (None, None), 1
        # In this case, dealer has hit >= 17, game ends
        # print("Game ended: let's compare")
        # print(player_sum, dealer_sum)
        if dealer_sum > player_sum:
             return (None, None), -1
        elif dealer_sum == player_sum:
             return (None, None), 0
        else:
             return (None, None), 1


def glie_mc_control(num_iter=1000):
    # Initialize count N(S, A) and value Q(S, A) table
    # action: 0 is stick, 1 is hit
    count_table = np.zeros((21, 10, 2))
    value_table = np.zeros((21, 10, 2))
    N_0 = 100
    cur_iter = 0
    while (cur_iter < num_iter):
        # We generate a new, full episode
        state_action_history = []
        initial_state = generate_initial_state()
        state_action_history.append(initial_state)
        # print("Initial state is", state_action_history)
        cur_state = initial_state
        # inner loop for getting to the end of an episode
        while(1):
            # pick an action with e-greedy strategy: e = N_0 / (N_0 + N(S))
            cur_state_count = count_table[cur_state[0]-1, cur_state[1]-1, :].sum()
            cur_eps = N_0 / (N_0 + cur_state_count)
            rng = random.random()
            if rng < cur_eps:
                action = random.randint(0, 1)
            elif value_table[cur_state[0]-1, cur_state[1]-1, 0] >= value_table[cur_state[0]-1, cur_state[1]-1, 1]:
                action = 0
            else:
                action = 1
            # take a step
            # print("Current action is", action)
            state_action_history.append(action)
            cur_state, cur_reward = step(cur_state, action)
            # check for termination
            if cur_state[0] is None:
                break
            else:
                state_action_history.append(cur_state)
                #  print("Current history is", state_action_history)
        # print("Final history is", state_action_history)
        # print(cur_reward)
        # Reward updates 
        for i in range(0, len(state_action_history), 2):
            count_table[state_action_history[i][0]-1, state_action_history[i][1]-1, state_action_history[i+1]] += 1
            cur_count = count_table[state_action_history[i][0]-1, state_action_history[i][1]-1, state_action_history[i+1]]
            value_table[state_action_history[i][0]-1, state_action_history[i][1]-1, state_action_history[i+1]] += \
                (cur_reward - value_table[state_action_history[i][0]-1, state_action_history[i][1]-1, state_action_history[i+1]]) / cur_count
        cur_iter += 1

    return count_table, value_table


def sarsa_lambda(num_iter=1000, lam=0.1):
    # Initialize count N(S, A), value Q(S, A)
    # action: 0 is stick, 1 is hit
    count_table = np.zeros((21, 10, 2))
    value_table = np.zeros((21, 10, 2))
    N_0 = 100
    cur_iter = 0
    while (cur_iter < num_iter):
        # We generate a new, full episode.
        # Different from MC, in here we want to perform updates after every time step
        # First, create the eligbility traces
        eligibility_table = np.zeros((21, 10, 2))
        initial_state = generate_initial_state()
        cur_state = initial_state
        # pick an action with e-greedy strategy: e = N_0 / (N_0 + N(S))
        cur_state_count = count_table[cur_state[0]-1, cur_state[1]-1, :].sum()
        cur_eps = N_0 / (N_0 + cur_state_count)
        rng = random.random()
        if rng < cur_eps:
            cur_action = random.randint(0, 1)
        elif value_table[cur_state[0]-1, cur_state[1]-1, 0] >= value_table[cur_state[0]-1, cur_state[1]-1, 1]:
            cur_action = 0
        else:
            cur_action = 1
        # print("Initial state is", initial_state)
        # print("initial action is", cur_action)
        while(1):
            count_table[cur_state[0]-1, cur_state[1]-1, cur_action] += 1
            next_state, cur_reward = step(cur_state, cur_action)
            # Case 1: non-terminal state
            if next_state[0] is not None:
                next_state_count = count_table[next_state[0]-1, next_state[1]-1, :].sum()
                cur_eps = N_0 / (N_0 + next_state_count)
                rng = random.random()
                if rng < cur_eps:
                    next_action = random.randint(0, 1)
                elif value_table[next_state[0]-1, next_state[1]-1, 0] >= value_table[next_state[0]-1, next_state[1]-1, 1]:
                    next_action = 0
                else:
                    next_action = 1
                next_predicted = value_table[next_state[0]-1, next_state[1]-1, next_action]
                td_error = cur_reward + next_predicted - value_table[cur_state[0]-1, cur_state[1]-1, cur_action]
            else:
                td_error = cur_reward - value_table[cur_state[0]-1, cur_state[1]-1, cur_action]
            eligibility_table[cur_state[0]-1, cur_state[1]-1, cur_action] += 1
            for i in range(21):
                for j in range(10):
                    for k in range(2):
                        if eligibility_table[i, j, k] > 0:
                            value_table[i, j, k] += td_error * eligibility_table[i, j, k] / count_table[i, j, k]
                            eligibility_table[i, j, k] *= lam
            if next_state[0] is not None:
                cur_state = next_state
                cur_action = next_action
            else:
                break
        cur_iter += 1

    return value_table


def sarsa_function(num_iter=1000, lam=0.1):
    # Initialize weights: we will use a total of 36 binary features: dealer [1,4],[4,7],[7,10]. buyer [1,6],[4,9],[7,12],[10,15],[13,18].[16,21]. action hit;stick
    # We also need a helper to map a state_action pair to the feature values X(S, A)
    def state_action_to_feature(state_in, action_in):
        player_sum, dealer_sum = state_in
        player_intervals = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
        dealer_intervals = [(1, 4), (4, 7), (7, 10)]
        feature_val = np.zeros((3, 6, 2))
        for player_idx, (player_lo, player_hi) in enumerate(player_intervals):
            for dealer_idx, (dealer_lo, dealer_hi) in enumerate(dealer_intervals):
                if player_lo <= player_sum <= player_hi and dealer_lo <= dealer_sum <= dealer_hi:
                    feature_val[dealer_idx, player_idx, action_in] = 1
        return feature_val
    weights = np.zeros((3, 6, 2))
    step_size = 0.01
    eps = 0.05
    cur_iter = 0
    while(cur_iter < num_iter):
        eligibility_scores = np.zeros((3, 6, 2))
        initial_state = generate_initial_state()
        cur_state = initial_state
        # pick an action with e-greedy strategy
        # we will enumerate all actions and fill out the corresponding feature values, then pick one that maximizes the q(S, A, W)
        rng = random.random()
        if rng < eps:
            cur_action = random.randint(0, 1)
        else:
            hit_feature = state_action_to_feature(cur_state, 1)
            stick_feature = state_action_to_feature(cur_state, 0)
            hit_value = np.sum(weights * hit_feature)
            stick_value = np.sum(weights * stick_feature)
            if hit_value >= stick_value:
                cur_action = 1
            else:
                cur_action = 0
        while(1):
            cur_feature = state_action_to_feature(cur_state, cur_action)
            cur_value = np.sum(weights * cur_feature)
            next_state, cur_reward = step(cur_state, cur_action)
            # Case 1: non-terminal state
            # We first generate the next action, then use that to compute TD error
            if next_state[0] is not None:
                rng = random.random()
                if rng < eps:
                    next_action = random.randint(0, 1)
                else:
                    hit_feature = state_action_to_feature(next_state, 1)
                    stick_feature = state_action_to_feature(next_state, 0)
                    hit_value = np.sum(weights * hit_feature)
                    stick_value = np.sum(weights * stick_feature)
                    if hit_value >= stick_value:
                        next_action = 1
                    else:
                        next_action = 0
                next_feature = state_action_to_feature(next_state, next_action)
                next_value = np.sum(weights * next_feature)
                td_error = cur_reward + next_value - cur_value
            # Case 2: terminal state
            else:
                td_error = cur_reward - cur_value
            # Now we perform eligiblity updates and weight updates
            eligibility_scores = eligibility_scores * lam + cur_feature
            weights += step_size * td_error * eligibility_scores
            if next_state[0] is not None:
                cur_state = next_state
                cur_action = next_action
            else:
                break
        cur_iter += 1
    # Let's actually calculate the value table so we can return it
    value_t = np.zeros((21, 10, 2))
    for i in range(21):
        for j in range(10):
            for k in range(2):
                cur_feature = state_action_to_feature((i+1, j+1), k)
                value_t[i, j, k] = np.sum(weights * cur_feature)
    return value_t


def get_MSE(mc_values, other_values):
    total_error = 0 
    for i in range(21):
        for j in range(10):
            for k in range(2):
                total_error += (mc_values[i, j, k] - other_values[i, j, k])**2
    return total_error / (21*10*2)


if __name__ == "__main__":
    test_MC = True
    test_sarsa = False
    test_function = True
    if test_MC:
        mc_count_t, mc_value_t = glie_mc_control(1000000)
        mc_policy_t = np.argmax(mc_value_t, axis=2)
        # print(policy_t)
    if test_sarsa:
        for cur_lam in np.arange(0, 1.1, 0.1):
            cur_value_t = sarsa_lambda(1000, cur_lam)
            policy_t = np.argmax(cur_value_t, axis=2)
            cur_error = get_MSE(mc_value_t, cur_value_t)
            print(f"Current lambda {cur_lam} has error {cur_error}")
        # print(policy_t)
    if test_function:
        for cur_lam in np.arange(0, 1.1, 0.1):
            cur_value_t = sarsa_function(1000, cur_lam)
            cur_error = get_MSE(mc_value_t, cur_value_t)
            print(f"Current lambda {cur_lam} has error {cur_error}")

    # X, Y = np.meshgrid(dealer_showing, player_sums)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(X, Y, policy_t, cmap="viridis")
    # ax.set_xlabel("Dealer showing")
    # ax.set_ylabel("Player sum")
    # ax.set_zlabel("Value")
    # plt.show()


