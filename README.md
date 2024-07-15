# HomotopyOpt.jl

This package solves a constrained optimization problem which minimizes an objective function restricted to an algebraic variety.
There are two main ideas. First, we use parameter homotopy (using `HomotopyContinuation.jl`) to attempt a line search in the direction of the projected gradient vector.
Because we use parameter homotopy, this *line search* is really a *curve search* along a curve that stays on the constraint variety.

Second, we use *parallel transport* to decide when to slow down and search more carefully. Whenever we observe that the norm of the projected
gradient has been decreasing and then starts to increase, we parallel transport a projected gradient from one tangent space to the other,
compute their dot product, and if it's negative, that means the projected gradient has *reversed direction*, so that we skipped past a critical point.
If this happens, we go back a bit, and slow down our search, looking more carefully in that neighborhood.
The end result is that we slow down in the correct places to find critical points where the projected gradient vector is essentially the zero vector.

## Installation

```
julia> ]
(@v1.9) pkg> add HomotopyOpt
```

## Usage

```julia
using HomotopyOpt

sexticcurve(x) = [(x[1]^4 + x[2]^4 - 1) * (x[1]^2 + x[2]^2 - 2) + x[1]^5 * x[2]] # sextic curve
N,d = 2,1 # ambient dimension, variety dimension
numsamples = 100 # we want to compute some random starting points for our optimization problem

G = ConstraintVariety(sexticcurve, N, d, numsamples); # if you dont ask for samples, it will not compute them.
```

Above we created a `ConstraintVariety`, and now we need to create a function that evaluates the gradient of the objective function.
For the objective function, we choose the squared distance from the point $(2,2)$ in the plane, for visualization purposes in this example.
```julia
Q = x->(x[1]-2)^2+(x[2]-2)^2
```

The main function is `findminima` which actually implements our algorithm. It takes inputs as follows:
```julia
p0 = rand(G.samples) # choose a random starting point on the curve
tolerance = 1e-3

result = findminima(p0, tolerance, G, Q);
```

Now we can `watch` our result.
```julia
watch(result)
```
which produces the following output:

![](https://github.com/matthiashimmelmann/HomotopyOpt.jl/blob/firstbranch/test/Images/watch1.661497754964e9.gif)
