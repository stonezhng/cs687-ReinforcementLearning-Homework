#Readme
## HOW TO RUN
Mean square td error calculation is in `TD.py`, run the script directly to draw the three curves. This file uses previously implemented grid_plus.py and cartpole.py to draw the curves. The step size ranges from $10^{-5}$ to 1 in a logarithmic scale.

## Main methods
* `td_grid(lrs)` calculates  mean square td errors with respect to every step size in the list `lrs`.
* `td_cp(lrs, f_order)` calculates  mean square td errors with respect to every step size in the list `lrs` using f_order Fourier base.
* `td_cp_single(f_order, alpha)
` is for test only, do not call it.
* `draw_multi_bar(x, y_map, limbase='nolim', filename='result.png')` draw all the three curves using the results calculated by above two methods. The default plot name is result.png, `limbase` is the limit of the y axis value. If using the default `limbase` value `nolim`, there will be no limit, and the plot will be sort of ugly and unclear (you cannot observe the fluctuations). `limbase` can receive string values like `cartpole5`, which the limit is set to be the 1.5 times mean square td error using step size = 0.1; or you can just use real numbers like 600, which will directly set the limit to that value.



