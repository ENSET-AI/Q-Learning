# Q-Learning

## Notebook Link

For the full implementation and execution, refer to the [Main.ipynb](./Main.ipynb) notebook.

## Introduction

Q-Learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy for a given finite Markov Decision Process (MDP). It is based on learning the Q-value, which represents the expected utility of taking a certain action in a given state and following the optimal policy thereafter.

---

## Summary

The Q-Learning algorithm iteratively updates Q-values for state-action pairs based on observed rewards and transitions. The agent explores the environment, learns from its experiences, and converges to an optimal policy.

### Key Features:

- **Environment Representation**: A grid graph where each state is a node, and edges represent possible actions.
- **Reward System**: Rewards are assigned to states to guide the agent toward the goal.
- **Q-Value Updates**: Q-values are updated using the Bellman equation.
- **Exploration Factor (Epsilon)**: Introduces randomness to improve exploration of the state space.

---

## Principle

The Q-Learning algorithm works as follows:

1. Initialize Q-values for all state-action pairs to zero.
2. For each step:
   - Observe the current state.
   - Choose an action based on the current Q-values (e.g., using an epsilon-greedy policy).
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

- $ Q(s_t, a_t) $: Current Q-value for state $ s_t $ and action $ a_t $.
- $ \alpha $: Learning rate, which determines how much new information overrides old information.
- $ R_t $: Reward received after taking action $ a_t $ in state $ s_t $.
- $ \gamma $: Discount factor, which determines the importance of future rewards.
- $ \max_{a} Q(s_{t+1}, a) $: Maximum Q-value for the next state $ s_{t+1} $ over all possible actions.

---

## Environment Map

The grid graph used in this implementation is represented as follows:

```python
grid_graph = {
    "1": [("2", "right"), ("5", "down")],
    "2": [("1", "left"), ("3", "right"), ("6", "down")],
    "3": [("2", "left"), ("4", "right"), ("7", "down")],
    "4": [("3", "left"), ("8", "down")],
    "5": [("1", "up"), ("6", "right"), ("9", "down")],
    "6": [("2", "up"), ("5", "left"), ("7", "right"), ("10", "down")],
    "7": [("3", "up"), ("6", "left"), ("8", "right"), ("11", "down")],
    "8": [("4", "up"), ("7", "left"), ("12", "down")],
    "9": [("5", "up"), ("10", "right"), ("13", "down")],
    "10": [("6", "up"), ("9", "left"), ("11", "right"), ("14", "down")],
    "11": [("7", "up"), ("10", "left"), ("12", "right"), ("15", "down")],
    "12": [("8", "up"), ("11", "left"), ("16", "down")],
    "13": [("9", "up"), ("14", "right")],
    "14": [("10", "up"), ("13", "left"), ("15", "right")],
    "15": [("11", "up"), ("14", "left"), ("16", "right")],
    "16": [("12", "up"), ("15", "left")],
}
```

---

## Code Snippets

### Q-Value Update Function

```python
# filepath: c:\Users\yahya\OneDrive\Documents\SDIA\S2\SMA-IAD\Q-Learning\Main.ipynb
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

### Epsilon-Greedy Action Selection

```python
# filepath: c:\Users\yahya\OneDrive\Documents\SDIA\S2\SMA-IAD\Q-Learning\Main.ipynb
def choose_action(current_state, neighbor_states, epsilon=0):
    if np.random.random() < epsilon:
        np.random.shuffle(neighbor_states)
        return neighbor_states[0]
    return get_next_state(current_state, neighbor_states)
```

---

## Execution Output

Below is an example of the execution output during the Q-Learning process. Only the first 5 steps and the last step are shown for brevity.

```
Step 1:
	current state: 1
	neighbor states: [('2', 'right'), ('5', 'down')]
	neighbor states Q values: [0, 0]
	Next action: "right" to state 2 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 2:
	current state: 2
	neighbor states: [('1', 'left'), ('3', 'right'), ('6', 'down')]
	neighbor states Q values: [0, 0, 0]
	Next action: "left" to state 1 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 3:
	current state: 1
	neighbor states: [('2', 'right'), ('5', 'down')]
	neighbor states Q values: [-1.0, 0]
	Next action: "down" to state 5 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 125:
	current state: 8
	neighbor states: [('4', 'up'), ('7', 'left'), ('12', 'down')]
	neighbor states Q values: [-1.5, -1.5, -1.0]
	Next action: "down" to state 12 with Q value = -1.0 and reward = -1
	Updates Q value: -1.5
Step 126:
	current state: 12
	neighbor states: [('8', 'up'), ('11', 'left'), ('16', 'down')]
	neighbor states Q values: [-1.0, -1.0, 0]
	Next action: "down" to state 16 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 127:
	current state: 16
Goal state found at 127 iteration
```

---

## Introduction epsilon, Hasard factor

The introduction of a luck/hasard factor epsilon made the learning much more efficient by randomly choosing a state to explore instead of following a deterministic approach.

### Execution Output

```
Step 1:
	current state: 1
	neighbor states: [('2', 'right'), ('5', 'down')]
	neighbor states Q values: [0, 0]
	Next action: "right" to state 2 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 2:
	current state: 2
	neighbor states: [('1', 'left'), ('3', 'right'), ('6', 'down')]
	neighbor states Q values: [0, 0, 0]
	Next action: "left" to state 1 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 3:
	current state: 1
	neighbor states: [('2', 'right'), ('5', 'down')]
	neighbor states Q values: [-1.0, 0]
	Next action: "down" to state 5 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 32:
	current state: 7
	neighbor states: [('3', 'up'), ('6', 'left'), ('8', 'right'), ('11', 'down')]
	neighbor states Q values: [-1.0, -1.0, 0, 0]
	Next action: "right" to state 8 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 33:
	current state: 8
	neighbor states: [('4', 'up'), ('12', 'down'), ('7', 'left')]
	neighbor states Q values: [-1.5, 0, 0]
	Next action: "down" to state 12 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 34:
	current state: 12
	neighbor states: [('8', 'up'), ('11', 'left'), ('16', 'down')]
	neighbor states Q values: [0, 0, 0]
	Next action: "down" to state 16 with Q value = 0 and reward = -1
	Updates Q value: -1.0
Step 35:
	current state: 16
Goal state found at 35 iteration
```

---
