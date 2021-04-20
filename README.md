# ConstrainedOptimizationByParameterHomotopy.jl

This collects code that solves a constrained optimization problem which minimizes an objective function restricted to an algebraic variety.
There are two main ideas. First, we use parameter homotopy (using `HomotopyContinuation.jl`) to attempt a line search in the direction of the projected gradient vector.
Because we use parameter homotopy, this *line search* is really a *curve search* along a curve that stays on the constraint variety.

Second, we use *parallel transport* to decide when to slow down and search more carefully. Whenever we observe that the norm of the projected
gradient has been decreasing and then starts to increase, we parallel transport a projected gradient from one tangent space to the other,
compute their dot product, and if it's negative, that means the projected gradient has *reversed direction*, so that we skipped past a critical point.
If this happens, we go back a bit, and slow down our search, looking more carefully in that neighborhood.
The end result is that we slow down in the correct places to find critical points where the projected gradient vector is essentially the zero vector.

```julia
include("maincode.jl")

xvarz = [Variable(:x,i) for i in 1:2] # using the symbolics from HomotopyContinuation.jl

sexticcurve(x,y) = (x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y # sextic curve
g1 = sexticcurve(xvarz...)
g = [g1] # list of defining equations for the constraint variety
N,d = 2,1 # ambient dimension, variety dimension
numsamples = 100 # we want to compute some random starting points for our optimization problem

G = ConstraintVariety(xvarz, g, N, d, numsamples); # if you dont ask for samples, it will not compute them.
```

Above we created a `ConstraintVariety`, and now we need to create a function that evaluates the gradient of the objective function.
For the objective function, we choose the squared distance from the point $(2,2)$ in the plane, for visualization purposes in this example.
Therefore the gradient can be computed by hand, and we hard-wire this gradient evaluation in a function below.
```julia
# let the objective function be squared distance from the point (2,2) in the plane
# minimizing this obj fcn will compute the closest point, or at least a locally closest point
evalgradobjfcn(p) = [2*(p[1] - 2), 2*(p[2] - 2)] # evaluates the gradient of the objective function
```

The main function is `findminima` which actually implements our algorithm. It takes some inputs as follows:
```julia
p0 = rand(G.samples) # choose a random starting point on the curve
initialstepsize = 0.01
tolerance = 1e-3

result = findminima(p0,initialstepsize,tolerance, G, evalgradobjfcn);
```

Now we can `watch` our result.
```julia
watch(result)
```
which produces the following output:

![](https://github.com/alexheaton2/ConstrainedOptimizationByParameterHomotopy.jl/blob/firstbranch/watch2021-04-20T11:29:41.721.gif)
