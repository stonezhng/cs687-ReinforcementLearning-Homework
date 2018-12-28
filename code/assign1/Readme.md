# Readme
## HOW TO RUN
Just open **grid.py** and run it. All three questions will run. If you want to run certain question, you need to commend the remaining part,

## Explanation for the code
### The Grid class
#### variables
* The states and actions are stored as lists.
* four lists are used to store those states that may hit the wall. During the running, given an action and a state, we check the corresponding list. If the state is in the list, then it means the action makes the agent hit the wall.
* pi_matrix is the optimal policy. Since I mainly rely on reasoning, I directly code this as a list.
* Gamma is set to 0.9

#### functions
* dynamic: using a random number to decide the actual action given the intended action
* P_and_R: calculating $S_{t+1}$ and $R_t$ given $S_t$ and $A_t$. It may be more beautiful to write these two calculations separately, but I do not want to store the dynamic result.
* pi_uniform: this is the function to simulate the policy that we uniform and random selection of actions.
* pi_optimized: this is the function using pi_matrix to simulate the optimal policy
* R: calculating $R_t$ given $S_t$ and $A_t$. It is aborted because it requires to store the dynamic result of action or use a seed to generate the random result.
* dzero: this is the function to simulate $d_0$. In our gridworld, it will always be 1.

### The Episode class
#### variables
* grid: just the grid mentioned above
* history: a chain of $S_t, A_t, S_{t+1}$
* rewards: a chain of $R_t$
* active: if active = 1, the episode is still not terminated. if active = 0 then otherwise.

####functions
* run_first_step: setting $S_0$ and store $S_0$ to history
* run_next_step_randomly: use the last element in history as $S_t$, use grid.pi_uniform function to decide $A_t$, use grid.P_and_R to compute $S_{t+1}$ and $R_t$, store $A_t$ and $S_{t+1}$ in history and $R_t$ in rewards
* run_next_step_with_optimization: same as above, except that grid.pi_optimized function is used to decide $A_t$.
* run_all_steps_randomly: run with run_next_step_randomly till the episode terminates.
* run_all_steps_with_optimization: run with run_next_step_with_optimization till the episode terminates.
* get_discount_reward: calculate $\sum(\gamma ^ t \cdot R_t)$ using rewards.

### Question 1
* We initialize grid, and use a for-loop to run 10,000 episodes with run_all_steps_randomly method.

### Question 2
* This is an example to verify whether $A_9 = 'AR'$ is better than $A_9 = 'AD'$ by simply changing the variable pi_matrix. Just as mentioned in the pdf, there are some states whose actions are not obvious through reasoning, and I just use this simulation to decide which action is the best or there exist multiple optimal actions.

### Question 3
* We initialize grid, and use a for-loop to run 10,000 episodes with run_all_steps_with_optimization method.

### Question 4
* We set $S_0 = 18$, run the code $100, 000$ times and count the number of times when $S_{11} = 21$. Since $A_t$ only depends on $S_t$, we do not need to simulate the entire process. Because we have to set $S_0 = 18$ and run the process fast, we  have to run the episode step by step using run_next_step_randomly method and break the loop when we find a valid history.

## Response
* TA gave full credit.

