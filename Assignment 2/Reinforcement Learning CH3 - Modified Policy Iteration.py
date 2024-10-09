from scipy.constants import value

# Slide Example for Value Iteration (RL-Course NTNU, Saeedvand)

states = 8 # number of states
A = ['l', 'r']  # actions
actions = 2

# In case that reward can be different to be in one state with different actions we candefine them seperately, in our example both
# left and rith actions lead to same reward in individual state [State, [State, action]] * in other examples can be different
#           S1     S2     S3       S4     S5        S6       S7       S8
Reward = [[0, 0],[2, 2],[1, 1],[-1, -1],[3, 3], [-3, -3], [-7, -7], [5, 5]]


TransitionProbability = [ #[Left, Right]
    #    S1          S2          S3          S4          S5          S6          S7          S8
    [[0.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # if you are in s1 the probability to go s2(L=0.7, R=0.3), and s3(L=0.3, R=0.7)
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # s2
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0]],  # s3
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # s4  (terminal) all zero
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7, 0.3], [0.3, 0.7]],  # s5
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],  # s6  100% to left
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # s7  (terminal)
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]   # s8  (terminal)
]

Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Initial Value estimation of each state (we can set random too)
gamma = 0.9 # discount factor
theta = 0.000001

def evaluate_policy(value, policy, terminal) -> list:
    while True:
        delta = 0
        for state in range(states):
            action = policy[state]
            v_old = value[state]
            # for s in range(states):
            #     v = v + TransitionProbability[state][s][action] * (Reward[s][action] + gamma * value[s])
            value[state] = round(sum([TransitionProbability[state][s][action] * (Reward[s][action] + gamma * value[s]) for s in range(states)]), 5)
            delta = max(delta, abs(v_old - value[state]))
        if delta < theta:
            print("Delta < Theta")
            break
    return value

def improve_policy(value, policy) -> list:
    new_policy = [-1] * len(policy)
    for i, p in enumerate(policy):
        new_policy[i] = p

    for state in range(states):
        old_action = policy[state]
        new_action = -1

        if old_action == 0:  # left action
           new_action = 1
        else: # right action
            new_action = 0

        old_value = sum(
            [TransitionProbability[state][s][old_action] * (Reward[s][old_action] + gamma * value[s]) for s in
             range(states)])
        new_value = sum(
            [TransitionProbability[state][s][new_action] * (Reward[s][new_action] + gamma * value[s]) for s in range(states)])

        if new_value > old_value:
            new_policy[state] = new_action

    return new_policy

def print_policy(policy, terminal, label = ""):
    printable_policy = [-1] * len(policy)
    for i in range(len(policy)):
        printable_policy[i] = "l" if policy[i] == 0 and policy[i] != 'T' else "r"

    for i, t in enumerate(terminal):
        if t == 'T':
            printable_policy[i] = t

    print(label, printable_policy)

# for iteration in range(0, 100):
#     NewValue = [0, 0, 0, 0, 0, 0, 0, 0] # Values of each state after one iteration
#     for i in range(states):
#         for a in range(actions):
#             value_temp = 0
#             for j in range(states):
#                 value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
#             NewValue[i] = round(max(NewValue[i], value_temp), 5)
#     bellman_factor = 0
#     for i in range(states):
#         bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
#     Value = NewValue
#     print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')' , sep=",    ")
#     if(bellman_factor < delta):
#         break

# Determine the policy (One time iteration)
NewValue = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
policy = [1] * 8 # left = 0, right = 1
Terminal = ['','','','T','', '', 'T', 'T']
# for i in range(states):
#     for a in range(actions):
#         value_temp = 0
#         for j in range(states):
#             value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
#         if(NewValue[i] < value_temp):
#             if(Terminal[i] != 'T'):
#                 policy[i] = A[a]
#                 NewValue[i] = max(NewValue[i], value_temp)
#             else:
#                 policy[i] = 'T'
iteration = 0
while True:
    iteration += 1
    print("")
    print("--------------------------------")
    print("Iteration:", iteration)


    Value = evaluate_policy(Value, policy, Terminal)
    print("Value: ", str(Value))

    new_policy = improve_policy(Value, policy)
    policy_change = False

    print_policy(policy, Terminal, "Old Policy: ")
    print_policy(new_policy, Terminal, "New Policy: ")

    for j in range(len(new_policy)):
        if policy[j] != new_policy[j]:
            policy_change = True
    policy = new_policy

    if policy_change == False:
        print("Terminate - Policy has converged. Breaking the loop.")
        break

print("--------------------------------")
print("Final Value Estimation: ", end="")
print(Value)
print()
print("The algoirthm's final policy is:", end="")
print_policy(policy, Terminal)
print("--------------------------------")

