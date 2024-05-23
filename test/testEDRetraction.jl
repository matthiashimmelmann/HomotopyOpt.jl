include("../src/Euclidean_distance_retraction_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random, BenchmarkTools, Plots

Random.seed!(1234);

printstyled("Double Parabola Test\n", color=:green)
@var x y
eqnz = (y-x^2-1)*(y+x^2+1)
p, v = [-2,5], [2,-5.5]
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y], eqnz, 2, 1)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
R_pV = [-0.09960472458061737, 1.0099211011587805]
global index = 0
m = []
euler_array = ["newton", "explicit", "midpoint", "heun" ]
for euler in 1:length(euler_array)
    global index = 0
    global method = euler_array[euler]
    while index <= 4
        try
            euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
            #@btime EDStep(index,method)
            linear_steps = []
            for _ in 1:1
                push!(linear_steps, EDStep(index,method)[2])
            end
            println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
            println("Solution: ", EDStep(index,method)[1])
            global index = index+1
            println("")
        catch e
            display(e)
            global index = index+1
        end
    end
end
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Average Linear Steps: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2])




println("\n")
printstyled("Sextic Test\n", color=:green)
f1 = (x^3-x*y^2+y+1)^2*(x^2+y^2-1)+y^2-5
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y],f1,2,1)
p = [0,-1.833333333333333333]
p,_ = Euclidean_distance_retraction_minimal.gaussnewtonstep([f1], differentiate([f1],[x,y])', [x,y], p; tol=1e-12)
v = [0 -1; 1 0]*evaluate(differentiate(G.equations, G.variables), G.variables=>p)
v = 2 .* v ./ v[1]
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
#display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))
euler_array = ["newton", "explicit"]
for euler in 1:length(euler_array)
    global index = 0
    global method = euler_array[euler]
    while index <= 4
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        @btime EDStep(index,method)
        linear_steps = []
        for _ in 1:5
            push!(linear_steps, EDStep(index,method)[2])
        end
        println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        println("Solution: ", EDStep(index,method)[1])
        global index = index+1
        println("")
    end
end
println("HC.jl")
#@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Average Linear Steps: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2])



#=
println("\n")
printstyled("Enneper Surface Test\n", color=:green)
@var x y z
eqnz = ((y^2-x^2)/2+2/9*z^3+2/3*z)^3 - 6*z*((y^2-x^2)/4-1/4*(x^2+y^2+8/9*z^2)*z+2/9*z)^2
p = [1,1,4.4785]
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y,z], [eqnz], 3, 2)
p,_ = Euclidean_distance_retraction_minimal.gaussnewtonstep([eqnz], differentiate([eqnz],[x,y,z])', [x,y,z], p; tol=1e-12)
xvarz = [x,y,z]
nlp = nullspace(evaluate(differentiate([eqnz], xvarz), xvarz=>p))
v1,v2 = 2.5*nlp[:,1],2.5*nlp[:,2]
EDStep = (i,t,s)->Euclidean_distance_retraction_minimal.EDStep(G, p, t*v1+s*v2; homotopyMethod="gaussnewton", amount_Euler_steps=i)
steps_list = []
dt, ds = -1:0.1:1, -1:0.1:1
heatmapdtds = zeros(length(dt), length(ds))
for i in 1:length(dt), j in 1:length(ds)
    #println("$((dt[i],ds[j]))")
    global index = -1
    linear_steps = []
    truepoint,_ = Euclidean_distance_retraction_minimal.EDStep(G, p, dt[i]*v1+ds[j]*v2; homotopyMethod="HomotopyContinuation")
    while index <= 6
        try
            res = EDStep(index,dt[i],ds[j])
            if isapprox.(res[1],truepoint)
                push!(linear_steps, res[2])
            end
        catch
            push!(linear_steps, 1000)
        end
        global index = index+1
    end
    heatmapdtds[i,j] = argmin(linear_steps)
end
display(heatmap(dt, ds, heatmapdtds))
=#


println("\n")
printstyled("Stiefel Manifold Test\n", color=:green)
#TODO CHECK THE ANSWER USING QR-decomp (also start point)
n = (6,6)
@var x[1:n[1],1:n[2]]
f3 = vcat(x*x' - LinearAlgebra.Diagonal([1 for _ in 1:n[1]])...)
f3 = rand(Float64, n[1]*n[2], Int(n[1]*(n[1]+1)/2))'*f3
xvarz = vcat([x[i,j] for i in 1:n[1], j in 1:n[2]]...)
global G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, f3, n[1]*n[2], n[1]*n[2]-Int(n[1]*(n[1]+1)/2))
p = filter(t -> norm(t)<250, [Euclidean_distance_retraction_minimal.gaussnewtonstep(f3, differentiate(f3,xvarz)', xvarz, 2 .* (rand(Float64,length(xvarz)) .- 0.5); tol=1e-14, factor=0.1)[1] for _ in 1:50])[1]
nlp = nullspace(evaluate(differentiate(f3, xvarz), xvarz=>p))
global v = real.(nlp[:,1] ./ (norm(nlp[:,1])))
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; euler_step=euler_step, homotopyMethod="gaussnewton", amount_Euler_steps=i)

euler_array = ["newton", "explicit"]
for euler in 1:length(euler_array)
    global index = 0
    global method = euler_array[euler]
    while index <= 4
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        @btime EDStep(index,method)
        linear_steps = []
        for _ in 1:5
            push!(linear_steps, EDStep(index,method)[2])
        end
        println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        println("Solution: ", EDStep(index,method)[1])
        global index = index+1
        println("")
    end
end
println("HC.jl")
#@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
sol = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[1]
println("Solution: ", sol)
println("Norm of the product T_pM * (p+v-R_p(v)): ", norm(nullspace(evaluate(differentiate(f3,xvarz), xvarz=>sol))'*((p+v)-sol)))
println("Average Linear Steps: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2])






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
p = toArray(p0)#+0.05*rand(Float64,9)
nlp = nullspace(evaluate(differentiate(barequations, xvarz), xvarz=>p))
v = 3 .* real.(nlp[:,1] ./ (norm(nlp[:,1])*nlp[1,1]))
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, barequations, 9, 1)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; euler_step=euler_step, homotopyMethod="gaussnewton", amount_Euler_steps=i)

euler_array = ["newton", "explicit"]
for euler in 1:length(euler_array)
    global index = 0
    global method = euler_array[euler]
    while index <= 4
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        @btime EDStep(index,method)
        linear_steps = []
        for _ in 1:5
            push!(linear_steps, EDStep(index,method)[2])
        end
        println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        println("Solution: ", EDStep(index,method)[1])
        global index = index+1
        println("")
    end
end
println("HC.jl")
#@btime(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
sol = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[1]
println("Solution: ", sol)
println("Norm of the product T_pM * (p+v-R_p(v)): ", norm(nullspace(evaluate(differentiate(barequations,xvarz), xvarz=>sol))'*((p+v)-sol)))
println("Average Linear Steps: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2])




#=
println("\n")
printstyled("Fisher retraction test\n", color=:green)
@var prob[1:50]
equations = [sum(prob[1:50]-1)]=#