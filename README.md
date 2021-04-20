# ConstrainedOptimizationByParameterHomotopy.jl

This collects code that solves a constrained optimization problem which minimizes an objective function restricted to an algebraic variety.
The main idea is to use parameter homotopy (using `HomotopyContinuation.jl`) to attempt a line search in the direction of the projected gradient vector.
Parallel transport is also used to decide when to slow down and search more carefully.
