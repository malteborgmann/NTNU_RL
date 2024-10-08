# Slide Example for Value Iteration (RL-Course NTNU, Saeedvand)

states = 8 # number of states
A = ['l', 'r', 's']  # actions
actions = 3

# In case that reward can be different to be in one state with different actions we candefine them seperately, in our example both
# left and rith actions lead to same reward in individual state [State, [State, action]] * in other examples can be different
#           S1     S2     S3       S4     S5        S6       S7       S8
Reward = [[0, 0, 0],[2, 2, 0],[1, 1, 0],[-1, -1, 0],[3, 3, 0], [-3, -3, 0], [-7, -7, 0], [5, 5, 0]]

# Added Stop action
# TransitionProbability has to be 1.0 for stop action, when state is recursing, and 0.0 for all other actions
# We choose the stop action if the V-Value is negative, so in consequence the V-Value is positive, the agent will choose to go to the left or right. So what the stop actions is to prevent from earning average negative rewards.
TransitionProbability = [ #[Left, Right, Stop]
    # S1               S2               S3               S4               S5               S6               S7               S8
    [[0.0, 0.0, 1.0], [0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # if you are in s1 the probability to go s2(L=0.7, R=0.3), and s3(L=0.3, R=0.7)
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # s2
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # s3
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # s4  (terminal) all zero
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.3, 0.7, 0.0]],  # s5
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # s6  100% to left
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],  # s7  (terminal)
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]   # s8  (terminal)
]

Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Initial Value estimation of each state (we can set random too)

#-------------------------------------

gamma = 0.9
bellman_factor = 0
delta = 0.001
NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]

for iteration in range(0, 100):
    NewValue = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
            NewValue[i] = round(max(NewValue[i], value_temp), 5)
    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
    Value = NewValue
    print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')' , sep=",    ")
    if(bellman_factor < delta):
        break

# Determine the policy (One time iteration)
NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
policy = ['NA','NA','NA','NA','NA', 'NA', 'NA', 'NA']
Terminal = ['','','','T','', '', 'T', 'T']
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
        if(NewValue[i] < value_temp):
            if(Terminal[i] != 'T'):
                policy[i] = A[a]
                NewValue[i] = max(NewValue[i], value_temp)
            else:
                policy[i] = 'T'

print("The algoirthm's final policy is:", policy)


# Back-Action
"""
The model for a back-action could be done in four different approaches:
1. Negative Reward
     - This model would lead to a penalty for taking the back-action.
     - This way we would prevent the agent from taking the back-action to often.
     - This could lead to a loop when previous V_pi is higher than the negative reward of the back_action
2. No Reward -> Reward of 0
     - This model would lead to the agent taking back-action for a kind of infinite reward trick.
     - For example when V_pi is negative
     - But this could lead the agent to be stuck in a loop. Switching from s2 to s5 and back would be a reward loop, because the reward for every second action is +3
3. Positive Reward
     - This model would lead to the agent taking the back-action more often.
     - We would get the same problem as in the No Reward -> Reward of 0 model.
     - Switching between two states endless would lead to a reward loop.
4. Negative Reward that is as high as the previous reward
     - This model would lead to the agent taking the back-action only when it is necessary.
     - The agent would only take the back-action when V_pi is lower than the previous V_pi.

In conclusion the best approach would be the fourth model. 
It must be noted that if the number of iterations is high enough the final policy would be pretty much the same as without a back-action.
"""


# Policy iteration
