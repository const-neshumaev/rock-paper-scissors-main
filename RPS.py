import numpy as np

state_dict = ["R", "P", "S"]

"""
indexes for each pair of an opponent and my plays
"""
indexes = {"RR":0, "RP":1, "RS":2, "PR":3, "PP":4, "PS":5, "SR":6, "SP":7, "SS":8}

WIN_REWARD = 1
TIE_REWARD = 0.3
LOST_REWARD = -0.1
STEP = 0.5
EPSILON = 0.4

rnd_gen = np.random.default_rng()

"""
I play like n-armed-bandit where arms are variants of previous game 
and I estimate my choise "R", "P" or "S" after that game 

estimates:
    rows - previous game (first positon is my play, second position is opponent play)
    columns - my play after previous game

            R  P  S
    RR    [[0. 0. 0.],
    RP     [0. 0. 0.],
    RS     [0. 0. 0.],
    PR     [0. 0. 0.],
    PP     [0. 0. 0.],
    PS     [0. 0. 0.],
    SR     [0. 0. 0.],
    SP     [0. 0. 0.],
    SS     [0. 0. 0.]]
"""
def player(prev_play, opponent_history=[], my_history=[], estimates=np.zeros((9,3), np.float64)):
    # first game (I play random)
    if prev_play == '': 
        guess = np.random.choice(state_dict)
        my_history.append(guess)
    # second game (I also play random and I don't need to estimate prev choise, because I did it randomly)
    elif len(opponent_history) == 0: 
        opponent_history.append(prev_play)
        guess = np.random.choice(state_dict)
        my_history.append(guess)
    else:
        opponent_history.append(prev_play)
        prev_prev_result = opponent_history[len(opponent_history) - 2] + my_history[len(my_history) - 2]
        prev_result = opponent_history[-1] + my_history[-1]
        if prev_result == "RP" or prev_result == "PS" or prev_result == "SR":
            reward = WIN_REWARD
        elif prev_result == "RR" or prev_result == "PP" or prev_result == "SS":
            reward = TIE_REWARD
        else:
            reward = LOST_REWARD

        row_ind = indexes.get(prev_prev_result)
        col_ind = state_dict.index(my_history[-1])
        
        # updating the average estimete of previous choise
        estimates[row_ind][col_ind] = estimates[row_ind][col_ind] + STEP * (reward - estimates[row_ind][col_ind])
        
        # chanse for exploration
        rnd = rnd_gen.random()

        # exploration
        if rnd <= EPSILON:
            guess = np.random.choice(state_dict)
        # exploitation
        else:
            row_ind = indexes.get(prev_result)
            guess = state_dict[np.argmax(estimates[row_ind])]
       
        my_history.append(guess)
        
    return guess
