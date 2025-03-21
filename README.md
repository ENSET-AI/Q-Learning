# Q-Learning

## Introduction

Q-Learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy for a given finite Markov Decision Process (MDP). It is based on learning the Q-value, which represents the expected utility of taking a certain action in a given state and following the optimal policy thereafter.

---

## Summary

The Q-Learning algorithm iteratively updates Q-values for state-action pairs based on observed rewards and transitions. The agent explores the environment, learns from its experiences, and converges to an optimal policy.

### Key Features:

- **Environment Representation**: A grid graph where each state is a node, and edges represent possible actions.
- **Reward System**: Rewards are assigned to states to guide the agent toward the goal.
- **Q-Value Updates**: Q-values are updated using the Bellman equation.

---

## Principle

The Q-Learning algorithm works as follows:

1. Initialize Q-values for all state-action pairs to zero.
2. For each step:
   - Observe the current state.
   - Choose an action based on the current Q-values (e.g., using a greedy policy).
   - Transition to the next state and observe the reward.
   - Update the Q-value for the current state-action pair using the Q-Learning update rule.
3. Repeat until the goal state is reached or a stopping condition is met.

---

## The Math

The Q-value update rule is given by:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( R_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right)
$$

Where:

- \( Q(s_t, a_t) \): Current Q-value for state \( s_t \) and action \( a_t \).
- \( \alpha \): Learning rate, which determines how much new information overrides old information.
- \( R_t \): Reward received after taking action \( a_t \) in state \( s_t \).
- \( \gamma \): Discount factor, which determines the importance of future rewards.
- \( \max*{a} Q(s*{t+1}, a) \): Maximum Q-value for the next state \( s\_{t+1} \) over all possible actions.

---

## Environment Map
The grid graph used in this implementation is represented as follows:

```python
grid_graph = {
    "1": [("2", "right"), ("6", "down")],
    "2": [("1", "left"), ("3", "right"), ("7", "down")],
    "3": [("2", "left"), ("4", "right"), ("8", "down")],
    "4": [("3", "left"), ("5", "right"), ("9", "down")],
    "5": [("4", "left"), ("10", "down")],

    "6": [("1", "up"), ("7", "right"), ("11", "down")],
    "7": [("2", "up"), ("6", "left"), ("8", "right"), ("12", "down")],
    "8": [("3", "up"), ("7", "left"), ("9", "right"), ("13", "down")],
    "9": [("4", "up"), ("8", "left"), ("10", "right"), ("14", "down")],
    "10": [("5", "up"), ("9", "left"), ("15", "down")],

    "11": [("6", "up"), ("12", "right"), ("16", "down")],
    "12": [("7", "up"), ("11", "left"), ("13", "right"), ("17", "down")],
    "13": [("8", "up"), ("12", "left"), ("14", "right"), ("18", "down")],
    "14": [("9", "up"), ("13", "left"), ("15", "right"), ("19", "down")],
    "15": [("10", "up"), ("14", "left"), ("20", "down")],

    "16": [("11", "up"), ("17", "right"), ("21", "down")],
    "17": [("12", "up"), ("16", "left"), ("18", "right"), ("22", "down")],
    "18": [("13", "up"), ("17", "left"), ("19", "right"), ("23", "down")],
    "19": [("14", "up"), ("18", "left"), ("20", "right"), ("24", "down")],
    "20": [("15", "up"), ("19", "left"), ("25", "down")],

    "21": [("16", "up"), ("22", "right")],
    "22": [("17", "up"), ("21", "left"), ("23", "right")],
    "23": [("18", "up"), ("22", "left"), ("24", "right")],
    "24": [("19", "up"), ("23", "left"), ("25", "right")],
    "25": [("20", "up"), ("24", "left")]
}
```
## Code Snippet

Here is a snippet of the Q-value update function from `Main.ipynb`:

```python
def get_Q_update(
    state_Q: float,
    state_reward: float,
    neighbor_states: list[str],
    learning_rate: float = 0.5,
    discount_factor: float = 0.5,
    step_penalty: float = 0
) -> float:
    neighbor_states_Q = map(lambda x: get_state_Q(*x), neighbor_states)
    return state_Q + learning_rate * (
        (state_reward + step_penalty) + discount_factor * max(neighbor_states_Q) - state_Q
    )
```

## Execution Output

Below is an example of the execution output during the Q-Learning process. Only the first 5 steps and the last step are shown for brevity.

```bash
Step 1:
	current state: 1
	neighbor states: [('2', 'right'), ('6', 'down')] 
	neighbor states Q values: [0, 0] 
	Next action: "right" to state 2 with Q value = 0 and reward = -1
	Updates Q value: -0.44999999999999996
Step 2:
	current state: 2
	neighbor states: [('1', 'left'), ('3', 'right'), ('7', 'down')] 
	neighbor states Q values: [0, 0, 0] 
	Next action: "left" to state 1 with Q value = 0 and reward = -1
	Updates Q value: -0.44999999999999996
Step 3:
	current state: 1
	neighbor states: [('2', 'right'), ('6', 'down')] 
	neighbor states Q values: [-0.44999999999999996, 0] 
	Next action: "down" to state 6 with Q value = 0 and reward = -1
	Updates Q value: -0.44999999999999996
Step 4:
	current state: 6
	neighbor states: [('1', 'up'), ('7', 'right'), ('11', 'down')] 
	neighbor states Q values: [0, 0, 0] 
	Next action: "up" to state 1 with Q value = 0 and reward = -1
	Updates Q value: -0.44999999999999996
Step 5:
	current state: 1
	neighbor states: [('2', 'right'), ('6', 'down')] 
	neighbor states Q values: [-0.44999999999999996, -0.44999999999999996] 
	Next action: "right" to state 2 with Q value = -0.44999999999999996 and reward = -1
	Updates Q value: -0.7649999999999999

Step 283:
	current state: 3
	neighbor states: [('2', 'left'), ('4', 'right'), ('8', 'down')] 
	neighbor states Q values: [-1.9604294999999998, -1.7180999999999997, -1.7180999999999997] 
	Next action: "right" to state 4 with Q value = -1.7180999999999997 and reward = -1
	Updates Q value: -1.9604294999999998
Step 284:
	current state: 4
	neighbor states: [('3', 'left'), ('5', 'right'), ('9', 'down')] 
	neighbor states Q values: [-1.247895, -1.13985, -1.13985] 
	Next action: "right" to state 5 with Q value = -1.13985 and reward = -1
	Updates Q value: -1.247895
Step 285:
	current state: 5
	neighbor states: [('4', 'left'), ('10', 'down')] 
	neighbor states Q values: [-1.43145, -1.107] 
	Next action: "down" to state 10 with Q value = -1.107 and reward = -1
	Updates Q value: -1.43145
Step 286:
	current state: 10
	neighbor states: [('5', 'up'), ('9', 'left'), ('15', 'down')] 
	neighbor states Q values: [-0.9854999999999999, -0.9854999999999999, -0.7649999999999999] 
	Next action: "down" to state 15 with Q value = -0.7649999999999999 and reward = -1
	Updates Q value: -0.9854999999999999
Step 287:
	current state: 15
	neighbor states: [('10', 'up'), ('14', 'left'), ('20', 'down')] 
	neighbor states Q values: [-0.7649999999999999, -0.7649999999999999, -0.44999999999999996] 
	Next action: "down" to state 20 with Q value = -0.44999999999999996 and reward = -1
	Updates Q value: -0.7649999999999999
Step 288:
	current state: 20
	neighbor states: [('15', 'up'), ('19', 'left'), ('25', 'down')] 
	neighbor states Q values: [-0.44999999999999996, -0.44999999999999996, 0] 
	Next action: "down" to state 25 with Q value = 0 and reward = -1
	Updates Q value: -0.44999999999999996
Step 289:
	current state: 25
Goal state found at 289 iteration

```

