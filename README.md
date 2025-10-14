
# 🃏 Blackjack Q-Learning Agent — Gymnasium Reinforcement Learning Tutorial

### 🎯 Overview

This repository demonstrates how to build, train, and visualize a **Q-learning agent** that learns to play the classic **Blackjack** game using **Gymnasium** (the modern successor to OpenAI Gym).
It serves as a hands-on introduction to **Reinforcement Learning (RL)** and **tabular Q-learning**, following the example from **Sutton & Barto’s** *Reinforcement Learning: An Introduction*.

---

### 📚  Objectives

* Interact with **Gymnasium environments**
* Implement a **Q-learning agent** from scratch
* Understand **epsilon-greedy exploration**
* Apply the **temporal difference (TD)** learning update rule
* Track and visualize agent performance over thousands of episodes
* Interpret **learned value functions** and **policies** for Blackjack

---

### 🧩 Environment Details

**Environment:** `Blackjack-v1` (from `gymnasium`’s `toy_text` suite)
**Observation (State):** A tuple

```
(player_sum, dealer_visible_card, usable_ace)
```

**Actions:**

| Action | Description               |
| :----: | :------------------------ |
|    0   | Stand (Stop taking cards) |
|    1   | Hit (Draw another card)   |

**Goal:** Achieve a hand sum greater than the dealer’s without exceeding 21.

---

### 🧠 Core Algorithm: Q-Learning

The repository implements **Q-learning**, a **model-free** reinforcement learning algorithm.
It learns an optimal policy by iteratively updating a *Q-table* that estimates the expected reward of taking each action in each state.

[
Q(s,a) \leftarrow Q(s,a) + \alpha , [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
]

Where:

* ( Q(s,a) ): value of taking action *a* in state *s*
* ( r ): received reward
* ( \alpha ): learning rate
* ( \gamma ): discount factor
* ( s' ): next state

---

### 🏗️ Repository Structure

```
blackjack-qlearning/
│
├── blackjack.ipynb     # Main Jupyter Notebook tutorial
├── README.md                     # Project description (this file)
├── requirements.txt               # Dependencies (gymnasium, numpy, matplotlib, seaborn, tqdm)
│
└── /static/img/                  # Visual assets for explanation (optional)
     ├── blackjack_AE_loop.jpg
     ├── blackjack_training_plots.png
     ├── blackjack_with_usable_ace.png
     └── blackjack_without_usable_ace.png
```

---

### ⚙️ Key Components

#### 1. **Agent Definition (`BlackjackAgent` class)**

Implements:

* **Q-table storage** using `defaultdict`
* **ε-greedy** action selection
* **Q-value update** using the TD rule
* **Epsilon decay** for exploration → exploitation transition
* **Training error tracking** for diagnostics

#### 2. **Training Loop**

Trains the agent over `n_episodes` (e.g., 100,000 games):

* Plays full Blackjack rounds using the agent’s current policy
* Updates Q-values after each action
* Decays ε to reduce exploration over time
* Records statistics using `gym.wrappers.RecordEpisodeStatistics`

#### 3. **Visualization**

Includes rolling averages and visual policy maps:

* **Training performance plots**

  * Episode rewards
  * Episode lengths
  * Training error trends
* **3D State-Value surfaces** and **heatmap policy visualizations**

  * Separate for “usable ace” and “no usable ace” cases

---

### 📈 Example Outputs

* **Training Curves:**
  Smooth convergence of rewards and decreasing TD errors.
* **Learned Policy Heatmaps:**
  The agent learns to *hit* when the player’s sum is low and *stand* when near 20–21.

---

### 🧮 Hyperparameters

| Parameter         | Meaning                        | Example                       |
| :---------------- | :----------------------------- | :---------------------------- |
| `learning_rate`   | How fast Q-values are updated  | 0.01                          |
| `discount_factor` | How much future rewards matter | 0.95                          |
| `epsilon`         | Exploration rate               | Starts at 1.0 → decays to 0.1 |
| `n_episodes`      | Number of training games       | 100,000                       |
