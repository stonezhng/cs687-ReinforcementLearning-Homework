# Readme

## How to run
### Question 1
* `sarsa_lambda.py` contains the script to plot sarsa lambda gridworld and sarsa lambda mountain car(Fourier).
* `qlearning_lambda.py` contains the script to plot qlearning lambda gridworld and qlearing lambda mountain car(Fourier).
* I do not include sarsa and qlearning implemented in the previous homework.
### Question 2
* `actor_critic.py` contains the script to plot actor-critic gridworld and actor-critic mountain car(Fourier).
### Question 3
* `reinforce.py` contains the script to plot REINFORCE gridworld without baseline and REINFORCE mountain car(Fourier) with baseline.

## Some notes
* `assignment_helper.py` stores and reads csv file. Due to the long computation time, when reusing the previous resuls for comparision instead of rerun the script and wait for hours, use `assignment_helper.py` to store the average rewards and std for future use.
* REINFORCE mountain car(Fourier) with baseline just cannot converge.