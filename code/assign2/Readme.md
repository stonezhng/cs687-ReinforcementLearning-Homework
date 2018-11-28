# Readme
* **grid_plus.py** contains basic classes for gridwolrd
* **cartpole.py** contains basic classes for cartpole
* **multiproc_cartpole_ce.py** using `execute_cartpole(trail_num, converge_count)` method to run cross-entropy method on cartpole problem. `converge_count` is set to decide within how many loops each trail has to terminate. Hyperparameters are set in `cartpole_sampling(theta, cm, K, Ke, N, epsilon)`.
* **multiproc_cartpole_fchc.py** using `execute_cartpole(trail_num, converge_count)` method to run fchc method on cartpole problem. `converge_count` is set to decide within how many loops each trail has to terminate. Hyperparameters are set in `cartpole_sampling(theta, sigma, N, bound=None)`. This cartpole_sampling method is actually running a complete trail.
* **multiproc_grid_ce.py** using `execute_grid(trail_num, converge_count) `method to run cross-entropy method on grid problem. `converge_count` is set to decide within how many loops each trail has to terminate. Hyperparameters are set in `grid_sampling(theta, cm, K, Ke, N, epsilon)`.
* **multiproc_grid_fchc.py** using `execute_grid(trail_num, converge_count) `method to run fchc method on grid problem. `converge_count` is set to decide within how many loops each trail has to terminate. Hyperparameters are set in `grid_params_sampling(theta, sigma, N, bound=None)`. This grid_params_sampling method is actually running a complete trail.
* `CORE_NUM` in four multiproc py is the number of multiprocesses. It is set to 4.  



