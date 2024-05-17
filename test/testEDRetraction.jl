include("../src/Euclidean_distance_retraction_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random, BenchmarkTools

Random.seed!(1234);

@var x y
eqnz = (y-x^2-1)*(y+x^2+1)
p, v = [-2,5], 6/4 .* [1,-4]
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y], eqnz, 2, 1)
EDStep = i->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", amount_Euler_steps=i)

@btime EDStep(0)
display(EDStep(0))
@btime EDStep(1)
display(EDStep(1))
@btime EDStep(2)
display(EDStep(2))
@btime EDStep(3)
display(EDStep(3))
@btime EDStep(4)
display(EDStep(4))
@btime EDStep(5)
display(EDStep(5))
@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
display(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))


f1 = (x^3-x*y^2+y+1)^2*(x^2+y^2-1)+y^2-5
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y],f1,2,1)
p = [0,-1.8333333]
v = [0 -1; 1 0]*evaluate(differentiate(G.equations, G.variables), G.variables=>p)
v = 2 .* v ./ v[1]
EDStep = i->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", amount_Euler_steps=i)
display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))

println("")
@btime EDStep(0)
display(EDStep(0))
@btime EDStep(1)
display(EDStep(1))
@btime EDStep(2)
display(EDStep(2))
@btime EDStep(3)
display(EDStep(3))
@btime EDStep(4)
display(EDStep(4))
@btime EDStep(5)
display(EDStep(5))
@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
display(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
