include("../src/Euclidean_distance_retraction_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random, BenchmarkTools

Random.seed!(1234);

printstyled("Double Parabola Test\n", color=:green)
@var x y
eqnz = (y-x^2-1)*(y+x^2+1)
p, v = [-2,5], 6/4 .* [1,-4]
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y], eqnz, 2, 1)
EDStep = i->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", amount_Euler_steps=i)

global index = 0
m = []
while index <= 5
    println("$(index) Euler Steps")
    display(median(@benchmark EDStep(index)))
    println("Solution: ", EDStep(index))
    global index = index+1
    println("")
end
println("HC.jl")
display(median(@benchmark(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))

println("\n")
printstyled("Sextic Test\n", color=:green)
f1 = (x^3-x*y^2+y+1)^2*(x^2+y^2-1)+y^2-5
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y],f1,2,1)
p = [0,-1.833333333333333333]
p = Euclidean_distance_retraction_minimal.gaussnewtonstep([f1], differentiate([f1],[x,y])', [x,y], p; tol=1e-12)
v = [0 -1; 1 0]*evaluate(differentiate(G.equations, G.variables), G.variables=>p)
v = 2 .* v ./ v[1]
EDStep = i->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", amount_Euler_steps=i)
#display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))

global index = 0
while index <= 5
    println("$(index) Euler Steps")
    @btime EDStep(index)
    println("Solution: ", EDStep(index))
    global index = index+1
    println("")
end
println("HC.jl")
@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))






println("\n")
printstyled("Stiefel Manifold Test\n", color=:green)

n = (5,5)
@var x[1:n[1],1:n[2]]
f3 = vcat(x*x' - LinearAlgebra.Diagonal([1 for _ in 1:n[1]])...)
xvarz = vcat([x[i,j] for i in 1:n[1], j in 1:n[2]]...)
global G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, f3, n[1]*n[2], n[1]*n[2]-Int(n[1]*(n[1]+1)/2))
p = vcat([LinearAlgebra.Diagonal([1 for _ in 1:n[1]])[i,j] for i in 1:n[1], j in 1:n[2]]...) + 0.001*randn(Float64,length(xvarz))
nlp = nullspace(evaluate(differentiate(f3, xvarz), xvarz=>p))
global v = real.(nlp[:,1] ./ (norm(nlp[:,1])))
EDStep = i->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", amount_Euler_steps=i)

global index = 0
while index <= 5
    println("$(index) Euler Steps")
    @btime EDStep(index)
    println("Solution: ", EDStep(index))
    global index = index+1
    println("")
end
println("HC.jl")
@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))






println("\n")
printstyled("Octahedron Test\n", color=:green)
function toArray(p)
    configuration = Vector{Float64}([])
    for i in 1:3, j in 4:6
        push!(configuration, p[i,j])
    end
    return configuration
end
@var x[1:3,1:6]
p0 = [0 0 -1.; 1 -1 0; 1 1 0; -1 -1 0; -1 1 0; 0 0 1;]'
edges = [(1,4), (1,5), (2,6), (3,6), (4,6), (5,6), (4,5), (2,5), (3,4)]
xs = zeros(HomotopyContinuation.Expression,(3,6))
xs[:,1:3] = p0[:,1:3]
xs[:,4:6] = x[:,4:6]
xvarz = vcat([x[i,j] for i in 1:3, j in 4:6]...)
barequations = [sum((xs[:,bar[1]]-xs[:,bar[2]]).^2) - sum((p0[:,bar[1]]-p0[:,bar[2]]).^2) for bar in edges]
barequations = Vector{Expression}(rand(Float64,8,9)*barequations)
global cursol = toArray(p0)#+0.05*rand(Float64,9)
nlp = nullspace(evaluate(differentiate(barequations, xvarz), xvarz=>cursol))
global v = 2.5 .* real.(nlp[:,1] ./ (norm(nlp[:,1])*nlp[1,1]))
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, barequations, 9, 1)
EDStep = i->Euclidean_distance_retraction_minimal.EDStep(G, cursol, v; homotopyMethod="gaussnewton", amount_Euler_steps=i)

global index = 0
while index <= 5
    println("$(index) Euler Steps")
    @btime EDStep(index)
    println("Solution: ", EDStep(index))
    global index = index+1
    println("")
end
println("HC.jl")
@btime(Euclidean_distance_retraction_minimal.EDStep(G, cursol, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, cursol, v; homotopyMethod="HomotopyContinuation"))

