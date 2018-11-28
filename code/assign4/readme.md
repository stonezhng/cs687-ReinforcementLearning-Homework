# Readme

## How to run
### Question 1 and 2
* `sarsa.py` contains the script to plot sarsa gridworld, sarsa cartpole(Fourier) and sarsa cartpole(rbf).
* `qlearning.py` contains the script to plot qlearning gridworld, qlearing cartpole(Fourier) and qlearning cartpole(rbf).
### Question 3
* `ce_grid.py` and `ce_cartpole.py` draw the plot of cem gridwold and cem cartpole. 
* Use previous results in Question 1 to draw the comparision plot by reading the csv file.
### Question 4
* `softmax/sarsa_softmax.py` draw the plot of sarsa softmax gridworld and sarsa softmax cartpole.
* `softmax/qlearning_softmax.py` draw the plot of qlearning softmax gridworld and qlearning softmax cartpole.
* Use previous results in Question 1 to draw the comparision plot by reading the csv file.
### Question 5
* `moutaincar/policy_improvement.py` draw the plot of sarsa mountain car and qlearning mountain car.

## Some notes
* `assignment_helper.py` stores and reads csv file. Due to the long computation time, when reusing the previous resuls for comparision instead of rerun the script and wait for hours, use `assignment_helper.py` to store the average rewards and std for future use.
* `function_approximation.py` contains all the phi, while `estimation.py` implements q function approximation, epsilon greedy and softmax. Sorry for the terrible naming, but I feel this assignment very torturous and have no spare time to correct it...
* `mountaincar/mountaincar.py` contains the simulation of mountain car 

