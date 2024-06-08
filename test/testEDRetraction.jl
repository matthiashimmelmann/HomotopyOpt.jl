include("../src/Euclidean_distance_retraction_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random, BenchmarkTools, Plots, ImplicitPlots, Manifolds

Random.seed!(1235);
testDict = Dict()
max_indices = 5
euler_array = ["newton", "explicit"]

function savetofile(testDict)
    open("Data/testEDResults2.txt", "w") do file
        for key in sort(collect(keys(testDict)))
            write(file, "$(key)\n")
            for key2 in keys(testDict[key])
                write(file, "$(key2)\n")
                for i in 1:length(testDict[key][key2])
                    if typeof(testDict[key][key2][i])==HomotopyContinuation.TrackerResult
                        write(file, "$(testDict[key][key2][i])")
                        continue
                    end
                    write(file, "$(i-1): $(testDict[key][key2][i][1])ms, $(testDict[key][key2][i][2])kb, $(testDict[key][key2][i][3]) linear solves\n")
                end
            end
            write(file, "\n")
        end
    end
end


function compute_medial_points(eqnz, vars, xy_vals; only_min=false)
    dg = differentiate(eqnz,vars)
    dH = differentiate(dg,vars)
    medial_points, curpoint, axis, axis2 = [], [], [], []
    for tup in xy_vals
        n = evaluate(dg, vars=>tup)
        n = n ./ norm(n)
        max_dist = norm(evaluate(dg, vars=>tup)) / opnorm(evaluate.(dH, vars=>tup))
        max_dist2 = 1/norm(evaluate.(dH, vars=>tup)*pinv(evaluate(dg, vars=>tup))')
        append!(medial_points, [tup+max_dist*n, tup-max_dist*n])
        push!(curpoint, tup)
        push!(axis, max_dist)
        push!(axis2, max_dist2)
        if isapprox(tup[1],0)
            display(max_dist2)
        end
    end
    mini, mini2 = argmin(axis), argmin(axis2)
    println("mini: ", axis[mini])
    println("minipoint: ", curpoint[mini])
    println("mini2: ", axis2[mini2])
    println("minipoint2: ", curpoint[mini2])
    if only_min
        medial_points=[]
        for tup in xy_vals
            n = evaluate(dg, vars=>tup)
            n = n ./ norm(n)
            max_dist = norm(evaluate(dg, vars=>tup)) / opnorm(evaluate.(dH, vars=>tup))
            append!(medial_points, [tup+mini*n, tup-mini*n])
        end
    end
    return medial_points
end


printstyled("Double Parabola Test\n", color=:green)
@var x y l
eqnz = (y-x^2-1)*(y+x^2+1)

p, v = [-2,5], 2*[1,-4]
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y], eqnz, 2, 1)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
R_pV = [0, 1]

Lagrange = eqnz*l + sum((p+v-[x,y]).^2)
relsols = [sol[1:2] for sol in real_solutions(HomotopyContinuation.solve(System(differentiate(Lagrange, [x,y,l]), variables=[x,y,l])))]

#display(nullspace(evaluate(differentiate(eqnz, [x,y]), [x,y]=>R_pV)')'*(R_pV-(p+v)))
plt = implicit_plot((u,w) -> (w-u^2-1)*(w+u^2+1); xlims=(-2.5,2.5), ylims=(-3.5,6.5), linewidth=5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=0.5, tickfontsize=16, labelfontsize=24, legend=false)
xy_vals = vcat([[t, t^2+1] for t in -2.5:0.00025:2.5], [[t, -t^2-1] for t in -2.5:0.00025:2.5])
medial_points = compute_medial_points(eqnz, [x,y], xy_vals)
scatter!(plt, [pt[1] for pt in medial_points], [pt[2] for pt in medial_points]; markersize=2, color=:black)
implicit_plot!(plt, (u,w) -> (w-u^2-1)*(w+u^2+1); xlims=(-2.5,2.5), ylims=(-3.5,6.5), linewidth=5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=0.5, tickfontsize=16, labelfontsize=24, legend=false)

foreach(sol->plot!(plt, [sol[1], (p+v)[1]], [sol[2], (p+v)[2]], arrow=false, color=:darkgrey, linewidth=5, label="", linestyle=:dot), relsols)
#plot!(plt, [Newtonstep[1], (p+v)[1]], [Newtonstep[2], (p+v)[2]], arrow=false, color=:darkgrey, linewidth=5, label="", linestyle=:dot)
plot!(plt, [p[1],p[1]+v[1]], [p[2],p[2]+v[2]], arrow=true, color=:green3, linewidth=6, label="")
foreach(sol->scatter!(plt, [sol[1]], [sol[2]]; color=:magenta, markersize=9), relsols)
scatter!(plt, [R_pV[1]], [R_pV[2]]; color=:red3, markersize=9)
scatter!(plt, [p[1]], [p[2]]; color=:black, markersize=9)

savefig(plt, "Images/DoubleParabolaTest.png")


global index = 0
m = []
euler_array = ["newton", "explicit"]
testDict["DoubleParabola"] = Dict()
for euler in 1:length(euler_array)
    testDict["DoubleParabola"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = mean(@benchmark EDStep(index,method))
        linear_steps = []
        push!(linear_steps, EDStep(index,method)[2])
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))

        sol = [EDStep(index,method)[1] for _ in 1:5]
        println("Solution: ", sol)
        global index = index+1
        if any(st->!isapprox(norm(st-R_pV), 0; atol=1e-4), sol)
            push!(testDict["DoubleParabola"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sum(linear_steps)/length(linear_steps)))"))
            continue
        end
        push!(testDict["DoubleParabola"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
    end
end
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation", print=true)
println("Average Linear Steps: ", )
if !isapprox(norm(Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation")[1]-R_pV), 0; atol=1e-6)
    testDict["DoubleParabola"]["HC.jl"] = [("x", "x", "x")]
else
    testDict["DoubleParabola"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
end
for key in keys(testDict)
    display(key)
    display(testDict[key])
end
savetofile(testDict)



println("\n")
printstyled("Sextic Test\n", color=:green)
f1 = (x^3-x*y^2+y+1)^2*(x^2+y^2-1)+y^2-5
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y],f1,2,1)
p = [0,-1.833333333333333333]
p,_ = Euclidean_distance_retraction_minimal.gaussnewtonstep([f1], differentiate([f1],[x,y])', [x,y], p; tol=1e-12)
display(p)
v = [0 -1; 1 0]*evaluate(differentiate(G.equations, G.variables), G.variables=>p)
v = 2 * v ./ v[1]
display(v)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
#display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))
euler_array = ["newton", "explicit"]
R_pV = [1.2290197279520099, -1.4279713341538283]

Lagrange = f1*l + sum((p+v-[x,y]).^2)
relsols = [[1.2290197279520096, -1.427971334153828],
[1.2897439969709807, 1.83539853859939],
[1.1551755060231654, 0.5390959458607674]]


#display(nullspace(evaluate(differentiate(eqnz, [x,y]), [x,y]=>R_pV)')'*(R_pV-(p+v)))
plt = implicit_plot((u,w) -> (u^3-u*w^2+w+1)^2*(u^2+w^2-1)+w^2-5; xlims=(-2.75,2.75), ylims=(-2.5,2.5), linewidth=5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=1, tickfontsize=16, labelfontsize=24, legend=false)
xy_vals = []
for t in -2.65:0.00025:2.65
    sols = real_solutions(solve(System(evaluate([f1], [x]=>[t]), variables=[y])))
    for sol in sols
        push!(xy_vals, [t,sol[1]])
    end
end

medial_points = compute_medial_points(f1, [x,y], xy_vals; only_min=false)
scatter!(plt, [pt[1] for pt in medial_points], [pt[2] for pt in medial_points]; markersize=2, color=:black)
foreach(sol->plot!(plt, [sol[1], (p+v)[1]], [sol[2], (p+v)[2]], arrow=false, color=:darkgrey, linewidth=5, label="", linestyle=:dot), relsols)
implicit_plot!(plt, (u,w) -> (u^3-u*w^2+w+1)^2*(u^2+w^2-1)+w^2-5; xlims=(-2.75,2.75), ylims=(-2.5,2.5), linewidth=5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=1, tickfontsize=16, labelfontsize=24, legend=false)

plot!(plt, [p[1],p[1]+v[1]], [p[2],p[2]+v[2]], arrow=true, color=:green3, linewidth=6, label="")
foreach(sol->scatter!(plt, [sol[1]], [sol[2]]; color=:magenta, markersize=9), relsols)
scatter!(plt, [R_pV[1]], [R_pV[2]]; color=:red3, markersize=9)
scatter!(plt, [p[1]], [p[2]]; color=:black, markersize=9)

savefig(plt, "Images/SexticTest.png")


testDict["Sextic"] = Dict()
for euler in 1:length(euler_array)
    testDict["Sextic"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        try
            euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
            local u = mean(@benchmark EDStep(index,method))
            linear_steps = []
            push!(linear_steps, EDStep(index,method)[2])
            #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
            sol = [EDStep(index,method)[1] for _ in 1:5]
            println("Solution: ", sol)
            global index = index+1
            if any(st->!isapprox(norm(st-R_pV), 0; atol=1e-4), sol)
                display("!")
                push!(testDict["Sextic"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sum(linear_steps)/length(linear_steps)))"))
                continue
            end
            push!(testDict["Sextic"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
        catch e
            display(e)
            push!(testDict["Sextic"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
            global index = index+1
        end
    end
end
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation")
if !isapprox(norm(Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation")[1]-R_pV), 0; atol=1e-6)
    testDict["Sextic"]["HC.jl"] = [("x", "x", "x")]
else
    testDict["Sextic"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
end
for key in keys(testDict)
    display(key)
    display(testDict[key])
end

savetofile(testDict)


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
xyz_vals = []
for t in -2:0.025:2, s in -2:0.025:2
    sols = real_solutions(solve(System(evaluate([eqnz], [x,y]=>[t,s]), variables=[z])))
    for sol in sols
        push!(xyz_vals, [t,s,sol[1]])
    end
end

medial_points = compute_medial_points(eqnz, [x,y,z], xyz_vals; only_min=false)
plt = scatter([p[1]], [p[2]], [p[3]])
scatter!(plt, [pt[1] for pt in medial_points], [pt[2] for pt in medial_points], [pt[3] for pt in medial_points]; markersize=2, color=:black)
#foreach(sol->plot!(plt, [sol[1], (p+v)[1]], [sol[2], (p+v)[2]], [sol[3], (p+v)[3]], arrow=false, color=:darkgrey, linewidth=5, label="", linestyle=:dot), relsols)
#implicit_plot!(plt, (u,w) -> (u^3-u*w^2+w+1)^2*(u^2+w^2-1)+w^2-5; xlims=(-2.15,2.15), ylims=(-2.15,2.15), linewidth=5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=1, tickfontsize=16, labelfontsize=24, legend=false)

#plot!(plt, [p[1],p[1]+v[1]], [p[2],p[2]+v[2]], [p[3],p[3]+v[3]], arrow=true, color=:green3, linewidth=6, label="")
#foreach(sol->scatter!(plt, [sol[1]], [sol[2]], [sol[3]]; color=:magenta, markersize=9), relsols)
#scatter!(plt, [R_pV[1]], [R_pV[2]], [R_pV[3]]; color=:red3, markersize=9)
scatter!(plt, [p[1]], [p[2]], [p[3]]; color=:black, markersize=9)

savefig(plt, "Images/EnneperSurface.png")


println("\n")
printstyled("Octahedron Test\n", color=:green)
testDict["Octahedron"] = Dict()

function toArray(p)
    return vcat([p[i,j] for i in 1:3, j in 4:6]...)
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
v = 10 .* real.(nlp[:,1] ./ (norm(nlp[:,1])*nlp[1,1]))
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, barequations, 9, 1)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; euler_step=euler_step, homotopyMethod="gaussnewton", amount_Euler_steps=i)
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation", print=true)
testDict["Octahedron"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
R_pV = Base.copy(linear_steps[1])

for euler in 1:length(euler_array)
    testDict["Octahedron"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = mean(@benchmark EDStep(index,method))
        linear_steps = []
        push!(linear_steps, EDStep(index,method)[2])
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        #println("Solution: ", EDStep(index,method)[1])
        global index = index+1
        if !isapprox(norm(EDStep(index,method)[1]-R_pV), 0; atol=1e-5)
            push!(testDict["Octahedron"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sum(linear_steps)/length(linear_steps)))"))
            continue
        end
        push!(testDict["Octahedron"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
    end
end
for key in keys(testDict)
    display(testDict[key])
end

savetofile(testDict)




println("\n")
euler_array = ["newton", "explicit"]
#TODO CHECK THE ANSWER USING QR-decomp (also start point)
for n in [(10,10)]
    testDict["Stiefel$(n)"] = Dict()
    printstyled("Stiefel Manifold Test $(n)\n", color=:green)
    @var x[1:n[1],1:n[2]]
    f3 = x*x' - LinearAlgebra.Diagonal([1 for _ in 1:n[1]])
    f3 = vcat([f3[i,i:end] for i in 1:size(f3)[1]]...)
    #f3 = rand(Float64, n[1]*n[1], Int(n[1]*(n[1]+1)/2))'*f3
    xvarz = vcat([x[i,j] for i in 1:n[1], j in 1:n[2]]...)
    global G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, f3, n[1]*n[2], n[1]*n[2]-Int(n[1]*(n[1]+1)/2))
    global M = Manifolds.Stiefel(n[1],n[2])
    
    global qrdecomp = svd(rand(Float64,n[1],n[2])).Vt
    global p = vcat([qrdecomp[i,j] for i in 1:n[1], j in 1:n[2]]...)
    nlp = nullspace(evaluate(differentiate(f3, xvarz), xvarz=>p))
    global v = 5 .* real.(nlp[:,1] ./ (norm(nlp[:,1])))
    global vStiefel = evaluate.(Matrix{Expression}(x), xvarz=>v)
    local u = mean(@benchmark retract(M, qrdecomp, vStiefel, CayleyRetraction()))
    testDict["Stiefel$(n)"]["CayleyRetraction"] = [(u.time/(1000*1000), u.memory/1000, 1)]
    local u = mean(@benchmark retract(M, qrdecomp, vStiefel, PadeRetraction(3)))
    testDict["Stiefel$(n)"]["PadeRetraction"] = [(u.time/(1000*1000), u.memory/1000, 1)]
    local u = mean(@benchmark retract(M, qrdecomp, vStiefel, PolarRetraction()))
    testDict["Stiefel$(n)"]["PolarRetraction"] = [(u.time/(1000*1000), u.memory/1000, 1)]
    local u = mean(@benchmark retract(M, qrdecomp, vStiefel, QRRetraction()))
    testDict["Stiefel$(n)"]["QRRetraction"] = [(u.time/(1000*1000), u.memory/1000, 1)]

    global ED_step_comp = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; euler_step=euler_step, homotopyMethod="gaussnewton", amount_Euler_steps=i)

    println("HC.jl")
    #@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
    local u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
    #println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
    linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation")
    #println("Average Linear Steps: ", )
    testDict["Stiefel$(n)"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
    global R_pV = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation")[1]

    for euler in 1:length(euler_array)
        testDict["Stiefel$(n)"][euler_array[euler]] = []
        global index = 0
        global method = euler_array[euler]
        while index <= max_indices
            euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
            local u = mean(@benchmark ED_step_comp(index,method))
            linear_steps = []
            push!(linear_steps, ED_step_comp(index,method)[2])
            #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
            #println("Solution: ", EDStep(index,method)[1])
            sol = ED_step_comp(index,method)[1]
            global index = index+1
            if !isapprox(norm(sol .- R_pV), 0; atol=1e-4)
                push!(testDict["Stiefel$(n)"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sum(linear_steps)/length(linear_steps)))"))
                continue
            end
            push!(testDict["Stiefel$(n)"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
        end
    end
    for key in keys(testDict)
        display(testDict[key])
    end
end

savetofile(testDict)


println("\n")
printstyled("Connelly Test\n", color=:green)
testDict["Connelly"] = Dict()

@var x[1:2,1:11]
xs = zeros(HomotopyContinuation.Expression,(2,11))
xs[:,:] = x
p0 = [0 0; 1 0; 1 1; 2 1; 3/2 1/2;
        3.0 1.0; 4 1; 5 1; 9/2 1/2; 5 0; 6 0]'
edges = [(1, 2), (2, 3), (2, 5), (3,4), (3,5), (4,5), (4,6), (5,9), (6,7), (7,8), (7,9), (8,9), (8,10), (9,10), (10,11)]
pinned_vertices = [1, 6, 11]
for pin in pinned_vertices
    xs[:,pin] = p0[:,pin]
end
freevertices = [2,3,4,5,7,8,9,10]
function toArray(q)
    return vcat([q[i,j] for i in 1:2, j in freevertices]...)
end

xvarz = vcat([x[i,j] for i in 1:2, j in freevertices]...)
barequations = [sum((xs[:,bar[1]]-xs[:,bar[2]]).^2) - sum((p0[:,bar[1]]-p0[:,bar[2]]).^2) for bar in edges]
N = 16 # ambient dimension
L0 = rand_subspace(N; codim = 1)
while true 
    try
        global R_L0 = solve(barequations; target_subspace = L0, variables=xvarz)
        break
    catch TruncationError
        continue
    end
end
Ω = solve(
    barequations,
    solutions(R_L0);
    variables = xvarz,
    start_subspace = L0,
    target_subspaces = [rand_subspace(N; codim = 1, real = true) for _ in 1:100],
    transform_result = (R,p) -> real_solutions(R),
    flatten = true
)
ps = [evaluate.(xs,xvarz=>Ωs) for Ωs in Ω]
display(ps[argmin([norm(toArray(t)-toArray(p0)) for t in ps])])
p = toArray(ps[argmin([norm(toArray(t)-toArray(p0)) for t in ps])])
display(p)


q = evaluate.(xs,xvarz=>p)
plt = plot([], []; xlims=(-0.5,6.5), ylims=(-0.5,1.5), linewidth=5, color=:steelblue, aspect_ratio=1.5, grid=false, label="", size=(800,800), tickfontsize=16)
for edge in edges
    plot!(plt, [q[1,edge[1]], q[1,edge[2]]], [q[2,edge[1]], q[2,edge[2]]], arrow=false, color=:steelblue, linewidth=4, label="")
end
for i in 1:size(q)[2]
    scatter!(plt, [q[1,i]], [q[2,i]], color=:black, mc=:black, markersize=7, label="")
end
savefig(plt, "Images/ConnellyTest.png")

nlp = nullspace(evaluate(differentiate(barequations, xvarz), xvarz=>p))
v = 10*nlp[:,1]
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, barequations, 16, 1)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; euler_step=euler_step, homotopyMethod="gaussnewton", amount_Euler_steps=i)
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")

u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation", print=true)
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation")[1])
testDict["Connelly"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
R_pV = Base.copy(linear_steps[1])
for euler in 1:length(euler_array)
    testDict["Connelly"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = mean(@benchmark EDStep(index,method))
        linear_steps = []
        push!(linear_steps, EDStep(index,method)[2])
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        println("Solution: ", EDStep(index,method)[1])
        if !isapprox(norm(EDStep(index,method)[1]-R_pV), 0; atol=1e-5)
            push!(testDict["Connelly"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sum(linear_steps)/length(linear_steps)))"))
        else
            push!(testDict["Connelly"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
        end
        q = evaluate.(xs, xvarz=>EDStep(index,method)[1])

        global index = index+1
        
        (index>3||euler_array[euler]!="newton") ? continue : nothing
        for edge in edges
            plot!(plt, [q[1,edge[1]], q[1,edge[2]]], [q[2,edge[1]], q[2,edge[2]]], arrow=false, color=:lightgrey, linewidth=4, label="")
        end
        for i in 1:size(q)[2]
            if !isapprox(norm(p0[:,i]-q[:,i]), 0; atol=1e-4)
                plot!(plt, [p0[1,i], q[1,i]], [p0[2,i], q[2,i]], arrow=true, color=:green3, linewidth=4, label="")
            end
        end
        for edge in edges
            plot!(plt, [p0[1,edge[1]], p0[1,edge[2]]], [p0[2,edge[1]], p0[2,edge[2]]], arrow=false, color=:steelblue, linewidth=4, label="")
        end
        for i in 1:size(p0)[2]
            scatter!(plt, [p0[1,i]], [p0[2,i]], color=:black, mc=:black, markersize=7, label="")
        end
        savefig(plt, "Images/ConnellyTest2.png")
    end
end
for key in keys(testDict["Connelly"])
    display(testDict["Connelly"][key])
end



savetofile(testDict)



println("\n")
printstyled("Gilbert Graph test\n", color=:green)
testDict["GilbertGraph"] = Dict()

function create_rigidity_matrix(p0,edges; torus=false)
    d, n = size(p0)[2], size(p0)[1]
    rig_mat = zeros(Float64,n*d,length(edges))
    for i in 1:n, j in 1:length(edges)
        if i in edges[j] && !torus
            rig_mat[d*(i-1)+1:d*(i-1)+d,j] = (i==edges[j][1]) ? p0[i,:]-p0[edges[j][2],:] : p0[i,:]-p0[edges[j][1],:]
        elseif i in edges[j] && torus
            rig_mat[d*(i-1)+1:d*(i-1)+d,j] = (i==edges[j][1]) ? p0[i,:]-p0[edges[j][2],:]+edges[j][3] : p0[i,:]-p0[edges[j][1],:]-edges[j][3]
        end
    end
    return(rig_mat)
end


#=Finds the independent edges in the given graph=#
function find_independent_edges(vxlist,edges,p0; torus=false)
    Jac_mat = create_rigidity_matrix(p0,edges; torus=torus)
    independent_edges=[]
    for i in 1:length(edges)
        if rank(create_rigidity_matrix(p0,edges[filter(j->j!=i,1:length(edges))]; torus=torus)) < rank(Jac_mat)
            push!(independent_edges, edges[i])
        end
    end
    return(independent_edges)
end


#= The main method which runs the test. We record the global rigidity of the subgraph H_3, where each vertex has at least degree 3,
in `H_global_rigidity`. We record the minimal radius such that the point cluster is rigid in `allrvals`. We record the quotients of 
#V(H_3)/#V(G) in `H_vals`. Finally, we record `numberHConfigs` configurations at n=`example_n` in `H_configs`.=#
function poisson_point_process(; r_fast_search_coefficient = 0.33, r_search_coefficient = 0.01, num_points=20, torus = false)
    allrvals, H_vals, H_global_rigidity, H_configs, edges = [], [], [], [], []
    global r_overshoot = 0.
    global xMin, xMax = -sqrt(num_points)/2, sqrt(num_points)/2;
    global yMin, yMax = -sqrt(num_points)/2, sqrt(num_points)/2;
    global xDelta, yDelta = xMax-xMin, yMax-yMin; #rectangle dimensions
    global areaTotal=xDelta*yDelta;

    global n_poiss =  num_points
    global xx, yy=xDelta*rand(n_poiss).+xMin, yDelta*(rand(n_poiss)).+yMin #x and y coordinates of Poisson points
    global p0 = hcat(xx,yy)
    #mod(marker,5000)==0 ? display(div(marker,5000)) : nothing


    for r in r_fast_search_coefficient:r_fast_search_coefficient:sqrt(num_points)*sqrt(2)
        global edges = []
        global edges = filter(t->(sum((p0[t[1],:]-p0[t[2],:]).^2) <= r^2), vcat([[(i,j) for j in i+1:n_poiss] for i in 1:n_poiss]...)) # Add edges for points closer than r
        global r_overshoot = r
        if ((rank(create_rigidity_matrix(p0,edges; torus=torus)) < size(p0)[1]*size(p0)[2]-3)) 
            continue # If the graph is not rigid, continue the search and increase r
        end
        break
    end

    for r in r_overshoot-r_fast_search_coefficient+r_search_coefficient:r_search_coefficient:r_overshoot+r_search_coefficient
        global edges = []
        global edges = filter(t->(sum((p0[t[1],:]-p0[t[2],:]).^2) <= r^2), vcat([[(i,j) for j in i+1:n_poiss] for i in 1:n_poiss]...)) # Add edges for points closer than r

        if ((rank(create_rigidity_matrix(p0,edges; torus=torus)) < size(p0)[1]*size(p0)[2]-3))
            continue # If the graph is not rigid, continue the search and increase r
        end
        #(r)
        global vxlist, edge_list = [i for i in 1:n_poiss], Base.copy(edges)
        global independent_edges = find_independent_edges(vxlist,edge_list,p0)
        #display(independent_edges)
        deleteat!(edge_list, findfirst(t->(independent_edges[end-1])==t, edge_list))

        display(rank(create_rigidity_matrix(p0,edges)) - (size(p0)[1]*size(p0)[2]-3))
        display(rank(create_rigidity_matrix(p0,edge_list)) - (size(p0)[1]*size(p0)[2]-3))
        break
    end

    return p0', edge_list
end

function toMatrix(p0, conf)
    q = zeros(Float64,(2,size(p0)[2]))
    q[:,1] = p0[:,1]
    q[1,2] = p0[1,2]
    q[2,2] = conf[1]
    count = 2
    for i in 1:2, j in 3:num_points
        q[i,j] = conf[count]
        count = count+1
    end
    return q
end

Random.seed!(1235);
num_points = 27
p0, edges = poisson_point_process(; num_points=num_points)
@var x[1:2, 1:num_points]

xs = zeros(HomotopyContinuation.Expression,(2,num_points))
xs[:,1] = p0[:,1]
xs[1,2] = p0[1,2]
xs[2,2] = x[2,2]
xs[:,3:num_points] = x[:,3:num_points]
xvarz = vcat(x[2,2], vcat([x[i,j] for i in 1:2, j in 3:num_points]...))
p = vcat(p0[2,2], vcat([p0[i,j] for i in 1:2, j in 3:num_points]...))

plt = plot([], []; xlims=(-sqrt(num_points)/1.9,sqrt(num_points)/1.9), ylims=(-sqrt(num_points)/1.9,sqrt(num_points)/1.9), linewidth=4, color=:steelblue, grid=false, label="", size=(800,800), tickfontsize=16)
for edge in edges
    plot!(plt, [p0[1,edge[1]], p0[1,edge[2]]], [p0[2,edge[1]], p0[2,edge[2]]], arrow=false, color=:steelblue, linewidth=4, label="")
end
for i in 1:size(p0)[2]
    scatter!(plt, [p0[1,i]], [p0[2,i]], color=:black, mc=:black, markersize=7, label="")
end
savefig(plt, "Images/GilbertGraphTest.png")

barequations = [sum((xs[:,bar[1]]-xs[:,bar[2]]).^2) - sum((p0[:,bar[1]]-p0[:,bar[2]]).^2) for bar in edges]
dg = nullspace(evaluate(differentiate(barequations, xvarz), xvarz=>p))
barequations = rand(Float64,num_points*2-4,length(barequations))*barequations
v = 1.5*Vector{Float64}(dg[:,1])
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz,barequations,num_points*2-3,1)
#=F = System(barequations, variables=xvarz)
display(F)
Euclidean_distance_retraction_minimal.HCnewtonstep(F,p)=#
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
#display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))
euler_array = ["newton", "explicit"]

testDict["GilbertGraph"] = Dict()
println("HC.jl")
linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation", print=true)
u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
testDict["GilbertGraph"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
R_pV = linear_steps[1]
#R_pV =  [-0.10386893199082876, -0.687083321227572, -1.0321262695081679, 1.9031210271801597, -2.1723757096171163, -1.224693817190252, 0.5315228325172509, 0.17936291043177147, 1.8893956605634585, 2.3604360777779685, 0.2561932309537942, 2.183698362528558, -0.9370980463177627, -1.1449954503298052, -1.8583678724375687, 2.1840458048027087, 0.09796204959477836, -2.1613798621192535, 0.49363119777358744, 2.071862189765919, -1.3152237268600644, -1.4052339734170436, -1.7690569104380207, -2.109432744710661, 0.18913729365551674, 0.4595141798011463, -1.831129643749599, 1.4753018779014155, -0.10512764494362918, 2.5561562580812263, -1.128112041399618, 2.186833916394935, 1.5540837424989766, -2.026472622869457, 1.999886161418336, 1.136828759026351, 1.1854874584197916, 0.02791704622818384, 1.6080227577396484, 1.326063130149908, -2.5188983904180757, -0.4021487936373933, -2.0387743611661215, -0.9495701782600748, 0.20809458732938638, -2.139311375829354, 0.22982108704292945, -2.628990572877142, -1.2526436187012342, -1.0389502741147554, 1.9895276310095587]
println("Solution: ", R_pV)

for euler in 1:length(euler_array)
    testDict["GilbertGraph"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = mean(@benchmark EDStep(index,method))
        
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        println("Solution: ", EDStep(index,method)[1])
        sol = EDStep(index,method)
        
        #=if !isapprox(norm(sol[1]-R_pV), 0; atol=1e-5)
            push!(testDict["GilbertGraph"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sol[2]))"))
        else=#
        push!(testDict["GilbertGraph"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sol[2]))
        
        global index = index+1
        
        index < max_indices ? continue : nothing 
        q = evaluate.(xs, xvarz=>sol[1])
        for edge in edges
            plot!(plt, [q[1,edge[1]], q[1,edge[2]]], [q[2,edge[1]], q[2,edge[2]]], arrow=false, color=:lightgrey, linewidth=4, label="")
        end
        for i in 1:size(q)[2]
            if !isapprox(norm(p0[:,i]-q[:,i]), 0; atol=1e-4)
                plot!(plt, [p0[1,i], q[1,i]], [p0[2,i], q[2,i]], arrow=true, color=:green3, linewidth=4, label="")
            end
        end
        for edge in edges
            plot!(plt, [p0[1,edge[1]], p0[1,edge[2]]], [p0[2,edge[1]], p0[2,edge[2]]], arrow=false, color=:steelblue, linewidth=4, label="")
        end
        for i in 1:size(p0)[2]
            scatter!(plt, [p0[1,i]], [p0[2,i]], color=:black, mc=:black, markersize=7, label="")
        end
        savefig(plt, "Images/GilbertGraphTest2.png")
    end
end
display(testDict)
savetofile(testDict)






println("\n")
printstyled("Sigma^+ test\n", color=:green)
testDict["Sigma^+"] = Dict()
p0 = [-0.375 -0.375 0.375; 0.625 -0.375 0.375; -0.375 0.625 0.375; -0.375 -0.375 1.375; 0.649084072506411 -0.14908407250641095 0.5581799812704914;
	0.43623724356957944 -0.15561862178478972 0.8443813782152103; 0.34438137821521025 0.06376275643042054 1.1556186217847897; 0.18623724356957944 0.09438137821521025 0.4056186217847897; 0.16830127018922195 0.08169872981077808 0.8316987298107781; 0.16830127018922195 -0.08169872981077808 1.081698729810778;
 	0.3277511116373524 -0.3277511116373524 0.5909399780242567; 0.34438137821521025 -0.06376275643042054 0.34438137821521025; 0.16830127018922195 -0.08169872981077808 0.6683012701892219; 0.4183012701892219 0.16830127018922192 0.6683012701892219; -0.3491240176901538 0.15087598230984625 0.9404665340384004;
	0.15841740970582857 0.34158259029417143 1.066762954598421; -0.16830127018922195 -0.16830127018922195 0.4183012701892219; -0.08169872981077805 0.3316987298107781 0.9183012701892219; 0.3287081698406671 0.3287081698406671 0.9115941651223108; -0.16617980971972543 0.16617980971972543 1.077740645974639;
	0.15561862178478972 -0.34438137821521025 0.43623724356957944; -0.3316987298107781 0.5816987298107781 0.9183012701892219; -0.40561862178478975 0.31376275643042056 1.0943813782152103; -0.125 0.625 0.625; -0.31376275643042056 0.40561862178478975 0.5943813782152103;
	-0.06376275643042055 0.6556186217847897 1.1556186217847897; -0.08169872981077808 0.3316987298107781 1.331698729810778; 0.625 0.125 0.875; -0.08169872981077805 -0.3316987298107781 0.5816987298107781; 0.625 0.375 0.625;
	0.625 -0.375 0.875; -0.40561862178478975 -0.31376275643042056 0.40561862178478975; 0.375 -0.375 0.625; 0.375 0.375 0.875; 0.125 -0.125 0.625;
	0.625 -0.125 0.625; 0.375 -0.125 0.875; 0.375 0.125 0.625; 0.375 0.125 1.125; 0.125 0.125 0.875;
	0.125 0.125 1.375; 0.125 -0.125 1.125; 0.125 0.375 1.125; -0.125 0.125 1.125; 0.125 -0.375 0.375;
	-0.125 -0.125 0.375; -0.375 0.375 1.125; -0.125 0.375 0.875; -0.125 0.625 1.125; -0.125 0.375 1.375;
	0.375 -0.125 0.375; 0.18623724356957944 0.09438137821521025 1.4056186217847897; 0.6508759823098462 0.15087598230984625 0.9404665340384004;  -0.125 -0.125 1.375; -0.350915927493589 -0.14908407250641095 0.5581799812704914;
	0.125 0.625 1.375; -0.375 0.625 1.375; -0.125 -0.375 0.625; -0.375 0.375 0.625; -0.375 0.625 0.875]

cables = [(24, 25), (22, 48), (30,34), (15, 23), (18, 20), (47,49), (26, 56), (57,50), (27,43), (16,40),
		(19, 39), (7, 10), (42, 54), (41, 44), (9, 37), (14, 28), (8, 12), (11, 51), (33, 31), (21, 35),
		(13, 38), (55, 32), (17, 29), (36,6),

		(6, 14), (7, 9), (52, 10), (16, 20), (21, 17), (23, 18), (26, 27), (12, 13), (53, 19), (5, 11),
		(1, 29), (25, 22), (6, 38), (14, 37), (37, 38), (7, 40), (9, 39), (39, 40), (52, 42), (10, 41),
		(41, 42), (16, 44), (20, 43), (43, 44), (21, 46), (17, 45), (45, 46), (23, 48), (18, 47), (47, 48),
		(26, 50), (27, 49), (49, 50), (12, 35), (13, 51), (51, 35), (53, 34), (19, 28), (28, 34), (5, 33),
		(11, 36), (36, 33), (1, 58), (32, 58), (25, 60), (22, 59), (59, 60), (29, 32)]

variablebars = [[([56,26], [26,49], [49,47]), ([57,50],[50,27],[27,43])], [([49,47], [47,23], [23,15]), ([22,48],[48,18],[18,20])], [([18,20], [20,44], [44,41]), ([27,43],[43,16],[16,40])],
				[([16,40], [40,9], [9,37]), ([10,7], [7,39], [39,19])], [([39,19], [19,34], [34,30]), ([14,28], [28,53], [15,23])], [([13,38], [38,14], [14,28]), ([9,37], [37,6], [6,36])],
				[([6,36], [36,5], [55,32]), ([31,33], [33,11], [11,51])], [([11,51], [51,12], [12,8]), ([38,13], [13,35], [35,21])], [([35,21], [21,45], [56,26]), ([29,17], [17,46], [54,42])],
				[([55,32], [32,1], [57,50]), ([25,24], [58,29], [29,17])], [([54,42], [42,10], [10,7]), ([44,41], [41,52], [8,12])], [([48,22], [22,60], [31,33]), ([24,25], [25,59], [30,34])]]
println("vertices")
foreach(i->println("$(i): ", p0[i,1], " ", p0[i,2], " ", p0[i,3]),1:size(p0)[1])
println("cables: ")
foreach(i->println("$(i): ", cables[i][1], " ",cables[i][2]),1:length(cables))
bars = vcat([varbar[1][2] for varbar in variablebars], [varbar[2][2] for varbar in variablebars])


tiedvertexinstructions = [ 	(52, 8, [0,0,1]),
							(53, 15, [1,0,0]),
							(54, 46, [0,0,1]),
							(55, 5, [-1,0,0]),
							(56, 45, [0,1,1]),
 							(57, 1, [0,1,1]),
							(58, 24, [0,-1,0]),
							(59, 30, [-1,0,0]),
                            (60, 31, [-1,1,0])]
#println([norm(p0[tis[1],:]-p0[tis[2],:]-tis[3][1]*(p0[2,:]-p0[1,:])-tis[3][2]*(p0[3,:]-p0[1,:])-tis[3][3]*(p0[4,:]-p0[1,:])) for tis in tiedvertexinstructions])
xs = Array{Any,2}(undef, size(p0))
xs[1:2,:] = p0[1:2,:]
xs[3,:] = [Variable(:x,3,1),Variable(:x,3,2),p0[3,3]]
xs[4,:] = [Variable(:x,4,1),Variable(:x,4,2),Variable(:x,4,3)]
for i in 5:51
    xs[i,:] = [Variable(:x,i,1), Variable(:x,i,2), Variable(:x,i,3)]
end
for instructions in tiedvertexinstructions
    vnew, vold, tran = instructions # unpacking
    c1,c2,c3 = tran # coefficients, how much of each lattice generator to use
    xs[vnew,:] = xs[vold,:] + c1*(xs[2,:]-xs[1,:]) + c2*(xs[3,:]-xs[1,:]) + c3*(xs[4,:]-xs[1,:])
end
display(xs)

function bardistance(varbar, xs)
	i,j,k = varbar
	cable1, base1 = xs[i[2],:]-xs[i[1],:], p0[i[2],:]-p0[i[1],:]
	cable2, base2 = xs[k[2],:]-xs[k[1],:], p0[k[2],:]-p0[k[1],:]
	straightbar = xs[j[2],:]-xs[j[1],:]
	dist = straightbar'*straightbar #This already is d^2
	r = 0.075
    return dist-r^2
	return (1-dist/(2*r^2))^2*(cable1'*cable1)*(cable2'*cable2) - (cable1'*cable2)^2
	#return dist*(1-r^2*dist*(cable1'*cable1))-r^2*(cable1'*straightbar)^2
end

function samePlane(varbar, xs)
	c1, c2 = xs[varbar[1][2],:]-xs[varbar[1][1],:], xs[varbar[4][2],:]-xs[varbar[4][1],:]
	b = xs[varbar[3],:]-xs[varbar[2],:]
	return(LinearAlgebra.cross(c1,c2)'*b)
end


# create `xvarz` to fix a variable ordering to pass to HC.jl solve functions etc.
# this ensures we know the meaning of the solution coordinates later after "solving"
xvarz = vcat([Variable(:x,3,1), Variable(:x,3,2)], [Variable(:x,i,k) for i in 4:51 for k in 1:3])

function toArray(q)
    return vcat([q[3,1],q[3,2]], [q[i,k] for i in 4:51 for k in 1:3])
end

function bar_equation(edge,xs)
    i,j = edge
    eqn = sum([(xs[i,k] - xs[j,k])^2 for k in 1:3])
    eqn += -sum([(p0[i,k] - p0[j,k])^2 for k in 1:3]) # constant term gives bar length^2 from p0
	return(eqn)
end

function medialbarequation(edge,xs,barlength)
	i, j = edge[1][2], edge[2][2]
	ci1,ci2, cj1,cj2 = edge[1][1],edge[1][3], edge[2][1],edge[2][3]
	m1, m2 = (xs[i[1],:]+xs[i[2],:])/2, (xs[j[1],:]+xs[j[2],:])/2
	bi, bj, mb = xs[i[2],:]-xs[i[1],:], xs[j[2],:]-xs[j[1],:], m1-m2
	cablei1, cablei2 = xs[ci1[2],:]-xs[ci1[1],:], xs[ci2[1],:]-xs[ci2[2],:]
	cablej1, cablej2 = xs[cj1[2],:]-xs[cj1[1],:], xs[cj2[1],:]-xs[cj2[2],:]
	bardist = [bardistance(edge[1], xs), bardistance(edge[2], xs)]
	sameplane = [LinearAlgebra.cross(cablei1,cablei2)'*bi, LinearAlgebra.cross(cablej1,cablej2)'*bj]
	sameorientation = [(cablei1'*mb)^2*(cablei2'*cablei2)-(cablei2'*mb)^2*(cablei1'*cablei1), (cablej1'*mb)^2*(cablej2'*cablej2)-(cablej2'*mb)^2*(cablej1'*cablej1)]
	centerbarlength = [mb'*mb-barlength^2]
	centerbarorth = [bi'*mb, bj'*mb]
	sameplanecenterbar = [LinearAlgebra.cross(cablei1,cablei2)'*mb, LinearAlgebra.cross(cablej1,cablej2)'*mb]
	#tetrahedralength = [cablei1'*cablei1-0.55^2, cablei2'*cablei2-0.55^2, cablej1'*cablej1-0.55^2, cablej2'*cablej2-0.55^2]
	#barsorth = [LinearAlgebra.cross(cablei1,cablei2)'*LinearAlgebra.cross(cablej1,cablej2)]
	#momentequations = make_moment_equations(mb, [(ci1,[ci2[2],ci2[1]]), (cj1,[cj2[2],cj2[1]])], xs)
	return(vcat(centerbarlength,centerbarorth,bardist,sameorientation,sameplane,sameplanecenterbar))
end

medialbar = [medialbarequation(edge, xs, 0.3535533905932738) for edge in variablebars]
collectmedialbars = Vector{Expression}([])
for bar in medialbar
	for entry in bar
		push!(collectmedialbars, entry)
	end
end
barequations = collectmedialbars

restinglength = 0.10 # something strictly less than 0.31754266665279735
constantofelasticity = 1.0
restinglengths = vcat([restinglength for cable in cables[1:24]], [0.3535533905932738 for cable in cables[25:end]])#, [0. for cable in cables[end-23:end]])
elasticitycoefficients = vcat([constantofelasticity for cable in cables[1:24]], [20. for cable in cables[25:end]])
cdict = Dict(vcat([cable => constantofelasticity for cable in cables[1:24]], [cable=>20. for cable in cables[25:end]])...)
rdict = Dict(vcat([cable => restinglength for cable in cables[1:24]], [cable => 0.3535533905932738 for cable in cables[25:end]])...)#,  [cable=>0. for cable in cables[end-23:end]])...)
function ambienttomatrix(sol)
    # returns a n by 3 array, the matrix of vertex coordinates
    # does it according to `tiedvertexinstructions` defined above
    point = zeros(typeof(sol[1]),size(p0)[1],3)
	point[1:4,:] = p0[1:4,:]
	point[3,1] = sol[1]
    point[3,2] = sol[2]
    counts = 3
    for i in 4:51
        for k in 1:3
            point[i,k] = sol[counts]
            counts += 1
        end
    end
    for instructions in tiedvertexinstructions
        vnew, vold, coeffs = instructions
        c1,c2,c3 = coeffs
        point[vnew,:] = point[vold,:] + c1*(point[2,:]-point[1,:]) + c2*(point[3,:]-point[1,:]) + c3*(point[4,:]-point[1,:])
    end
    return point
end

p = [-0.3718826856895793, 0.6115672116831886, -0.3724206537849344, -0.3750474712345589, 1.3616023026956794, 0.6072952517962481, -0.09281281304606669, 0.6191461590505964, 0.38548011044819797, -0.10537895174382929, 0.8798164058007497, 0.32996903579487474, 0.11769025764695838, 1.161499479768331, 0.1342696408853391, 0.1447176935957022, 0.41323581440685236, 0.12824867248108227, 0.11824456413121312, 0.8684974859517754, 0.13668244419081513, -0.07875311609337338, 1.1232574408513318, 0.34623624425564536, -0.3344455602965867, 0.6196555248139076, 0.3275314254873193, -0.05236350384016012, 0.3748973328393271, 0.12647536638153167, -0.051849961012777314, 0.668376548003521, 0.38556233110887644, 0.17135929815998505, 0.6565865438615873, -0.3903891477124731, 0.15854889313200063, 0.9169895206528939, 0.1103779079179276, 0.4004420624715116, 1.1125014651477765, -0.15413127258214607, -0.092517357231821, 0.4240692258087932, -0.1397604326435707, 0.3771089536239719, 0.8939045998407517, 0.34893935345259686, 0.4006326596431405, 0.9175307568254035, -0.15072700948605086, 0.15885345545621035, 1.1129054269891692, 0.10645761197163146, -0.3346914743772608, 0.42364703790321956, -0.3629401952745447, 0.6381637744030789, 0.9065523548090276, -0.4004587843997883, 0.3776591592757548, 1.1359553528916337, -0.1700916564977666, 0.6111174223202601, 0.6679545681363872, -0.36317024534274867, 0.41465639145047833, 0.6300365849321174, -0.1117153124831406, 0.6645739313360374, 1.1497486501535314, -0.11180570291897472, 0.38797580622368105, 1.3731497791036424, 0.5796445152128457, 0.13284545334284903, 0.8532215653048232, -0.14342419106783474, -0.3115883658042752, 0.642272170680595, 0.5719682384167806, 0.3880696122544055, 0.6567035490770956, 0.5690592574829707, -0.3217862109034212, 0.8799687221018129, -0.4045831691464021, -0.31106798652553486, 0.40073973332061913, 0.3759916980078119, -0.36009166940391096, 0.6835451970180076, 0.3787058084226504, 0.4264061516253418, 0.8536974950230768, 0.09650707198396528, -0.11561289213560276, 0.6426629293442727, 0.5777414521195001, -0.06711505722246752, 0.6831086362925969, 0.3206938427049997, -0.07867799858728997, 0.9065521289819328, 0.320589775684363, 0.14488251879607522, 0.6300809353149706, 0.3597672072793816, 0.1815562977378124, 1.1358441101286683, 0.09867742695864022, 0.18218161578472286, 0.8942384059966059, 0.07198507704134652, 0.1713165524126276, 1.373236492177969, 0.07177561023943256, -0.10529829585381317, 1.1498554981165154, 0.08077252832466468, 0.42616736839422115, 1.1764290038879, -0.12097852674810837, 0.13318720240048737, 1.1767902561530534, 0.07650358226495457, -0.3603908239264864, 0.35987165248344627, -0.1243226819965555, -0.06675207243063486, 0.3602523147238122, -0.3705193136483345, 0.4414463232439339, 1.1616424311092168, -0.16950610314544245, 0.4409539314218621, 0.8681361323605602, -0.17667292248649308, 0.6380674207613051, 1.1232361268862296, -0.1766231332716929, 0.4146585913816646, 1.3998280436283375, 0.35726440655667857, -0.11621146966716538, 0.4006730410316082]
p,_ = Euclidean_distance_retraction_minimal.gaussnewtonstep(barequations, differentiate(barequations,xvarz)', xvarz, p; tol=1e-15)
dg = nullspace(evaluate(differentiate(barequations,xvarz), xvarz=>p); atol=1e-6)
display(dg)
#barequations = rand(Float64,length(xvarz)-size(dg)[2],length(barequations))*barequations
#barequations = barequations[1:(length(xvarz)-size(dg)[2])]
v = Vector{Float64}(dg[:,1])
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz,barequations,length(xvarz),size(dg)[2])
display(G)
display(rank(evaluate.(G.jacobian,xvarz=>rand(Float64, length(xvarz)))))
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)

println("HC.jl")
linear_steps = Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation", print=true)
u = mean(@benchmark Euclidean_distance_retraction_minimal.EDStep_HC(G, p, v; homotopyMethod="HomotopyContinuation"))
testDict["Sigma^+"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, linear_steps[2]), linear_steps[3]]
R_pV = linear_steps[1]
println("Solution: ", R_pV)

for euler in 1:length(euler_array)
    testDict["Sigma^+"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = mean(@benchmark EDStep(index,method))
        
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        println("Solution: ", EDStep(index,method)[1])
        sol = EDStep(index,method)
        
        if !isapprox(norm(sol[1]-R_pV), 0; atol=1e-5)
            push!(testDict["Sigma^+"][euler_array[euler]], ("x ($(u.time/(1000*1000)))", "x ($(u.memory/1000))", "x ($(sol[2]))"))
        else
            push!(testDict["Sigma^+"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sol[2]))
        end
        
        global index = index+1
    end
end
display(testDict)

savetofile(testDict)