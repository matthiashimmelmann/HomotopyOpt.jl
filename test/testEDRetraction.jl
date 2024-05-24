include("../src/Euclidean_distance_retraction_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random, BenchmarkTools, Plots, ImplicitPlots

Random.seed!(1235);
testDict = Dict()
max_indices = 4

printstyled("Double Parabola Test\n", color=:green)
@var x y
eqnz = (y-x^2-1)*(y+x^2+1)


p, v = [-2,5], 5.5/4*[1,-4]
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y], eqnz, 2, 1)
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
R_pV = [-0.15440926990518755, 1.0238422226326531]

#display(nullspace(evaluate(differentiate(eqnz, [x,y]), [x,y]=>R_pV)')'*(R_pV-(p+v)))
plt = implicit_plot((u,w) -> (w-u^2-1)*(w+u^2+1); xlims=(-2.5,2.5), ylims=(-3.5,6.5), linewidth=7.5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=0.5, tickfontsize=16, labelfontsize=24)
plot!(plt, [R_pV[1], (p+v)[1]], [R_pV[2], (p+v)[2]], arrow=false, color=:darkgrey, linewidth=5, label="", linestyle=:dot)
plot!(plt, [p[1],p[1]+v[1]], [p[2],p[2]+v[2]], arrow=true, color=:green3, linewidth=7.5, label="")
savefig(plt, "DoubleParabolaTest.png")

global index = 0
m = []
euler_array = ["newton", "explicit"]
testDict["DoubleParabola"] = Dict()
for euler in 1:length(euler_array)
    testDict["DoubleParabola"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        try
            euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
            local u = median(@benchmark EDStep(index,method))
            linear_steps = []
            for _ in 1:5
                push!(linear_steps, EDStep(index,method)[2])
            end
            #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
            #println("Solution: ", EDStep(index,method)[1])
            global index = index+1
            if !isapprox(norm(EDStep(index,method)[1]-R_pV), 0; atol=1e-6)
                push!(testDict["DoubleParabola"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
                continue
            end
            push!(testDict["DoubleParabola"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
        catch e
            display(e)
            push!(testDict["DoubleParabola"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
            global index = index+1
        end
    end
end
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
u = median(@benchmark Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2]
println("Average Linear Steps: ", )
if !isapprox(norm(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[1]-R_pV), 0)
    testDict["DoubleParabola"]["HC.jl"] = [("ERROR", "ERROR", "ERROR")]
else
    testDict["DoubleParabola"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps))]
end
display(testDict)




println("\n")
printstyled("Sextic Test\n", color=:green)
f1 = (x^3-x*y^2+y+1)^2*(x^2+y^2-1)+y^2-5
G = Euclidean_distance_retraction_minimal.ConstraintVariety([x,y],f1,2,1)
p = [0,-1.833333333333333333]
p,_ = Euclidean_distance_retraction_minimal.gaussnewtonstep([f1], differentiate([f1],[x,y])', [x,y], p; tol=1e-12)
v = [0 -1; 1 0]*evaluate(differentiate(G.equations, G.variables), G.variables=>p)
v = v ./ v[1]
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
#display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))
euler_array = ["newton", "explicit"]
R_pV = [0.9654471100032206, -1.3760897428208472]

plt = implicit_plot((u,w) -> (u^3-u*w^2+w+1)^2*(u^2+w^2-1)+w^2-5; xlims=(-3,2.5), ylims=(-2.5,3), linewidth=7.5, color=:steelblue, grid=false, label="", size=(800,800), aspect_ratio=0.9, tickfontsize=16, labelfontsize=24)
plot!(plt, [R_pV[1], (p+v)[1]], [R_pV[2], (p+v)[2]], arrow=false, color=:darkgrey, linewidth=5, label="", linestyle=:dot)
plot!(plt, [p[1],p[1]+v[1]], [p[2],p[2]+v[2]], arrow=true, color=:green3, linewidth=7.5, label="")
savefig(plt, "SexticTest.png")

testDict["Sextic"] = Dict()
for euler in 1:length(euler_array)
    testDict["Sextic"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        try
            euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
            local u = median(@benchmark EDStep(index,method))
            linear_steps = []
            for _ in 1:5
                push!(linear_steps, EDStep(index,method)[2])
            end
            #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
            #println("Solution: ", EDStep(index,method)[1])
            global index = index+1
            if !isapprox(norm(EDStep(index,method)[1]-R_pV), 0)
                push!(testDict["Sextic"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
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
u = median(@benchmark Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2]
println("Average Linear Steps: ", )
if !isapprox(norm(Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[1]-R_pV), 0; atol=1e-6)
    testDict["Sextic"]["HC.jl"] = [("ERROR", "ERROR", "ERROR")]
else
    testDict["Sextic"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps))]
end
display(testDict)



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
#TODO CHECK THE ANSWER USING QR-decomp (also start point)
for n in [(4,7),(5,7),(6,7),(7,7)]
    printstyled("Stiefel Manifold Test $(n)\n", color=:green)
    @var x[1:n[1],1:n[2]]
    f3 = vcat(x*x' - LinearAlgebra.Diagonal([1 for _ in 1:n[1]])...)
    f3 = rand(Float64, n[1]*n[1], Int(n[1]*(n[1]+1)/2))'*f3
    xvarz = vcat([x[i,j] for i in 1:n[1], j in 1:n[2]]...)
    global G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz, f3, n[1]*n[2], n[1]*n[2]-Int(n[1]*(n[1]+1)/2))
    qrdecomp = svd(rand(Float64,n[1],n[2])).Vt
    global p = vcat([qrdecomp[i,j] for i in 1:n[1], j in 1:n[2]]...)
    nlp = nullspace(evaluate(differentiate(f3, xvarz), xvarz=>p))
    global v = real.(nlp[:,1] ./ (norm(nlp[:,1])))
    EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; euler_step=euler_step, homotopyMethod="gaussnewton", amount_Euler_steps=i)

    testDict["Stiefel$(n)"] = Dict()
    println("HC.jl")
    #@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
    local u = median(@benchmark Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
    println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
    linear_steps = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2]
    println("Average Linear Steps: ", )
    testDict["Stiefel$(n)"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps))]

    global R_pV = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[1]
    for euler in 1:length(euler_array)
        testDict["Stiefel$(n)"][euler_array[euler]] = []
        global index = 0
        global method = euler_array[euler]
        while index <= max_indices
            try
                euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
                local u = median(@benchmark EDStep(index,method))
                linear_steps = []
                for _ in 1:5
                    push!(linear_steps, EDStep(index,method)[2])
                end
                #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
                #println("Solution: ", EDStep(index,method)[1])
                global index = index+1
                if !isapprox(norm(EDStep(index,method)[1] .- R_pV), 0; atol=1e-6)
                    push!(testDict["Stiefel$(n)"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
                    continue
                end
                push!(testDict["Stiefel$(n)"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
            catch e
                display(e)
                push!(testDict["Stiefel$(n)"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
                global index = index+1
            end
        end
    end
    display(testDict)
end


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
R_pV = [-0.04389030151885144, -1.6052196028741612, -0.35089009560698886, -1.6442303344309768, -0.5330538377295431, -0.8888235032985654, -0.40099064822188346, -2.0216649379093323e-16, 0.19293834143276115]

testDict["Octahedron"] = Dict()
for euler in 1:length(euler_array)
    testDict["Octahedron"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = median(@benchmark EDStep(index,method))
        linear_steps = []
        for _ in 1:5
            push!(linear_steps, EDStep(index,method)[2])
        end
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        #println("Solution: ", EDStep(index,method)[1])
        global index = index+1
        if !isapprox(norm(EDStep(index,method)[1]-R_pV), 0; atol=1e-6)
            push!(testDict["Octahedron"][euler_array[euler]], ("ERROR", "ERROR", "ERROR"))
            continue
        end
        push!(testDict["Octahedron"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps)))
end
end
println("HC.jl")
#@btime Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")
u = median(@benchmark Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2]
println("Average Linear Steps: ", )
testDict["Octahedron"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps))]
display(testDict)

=#


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
        display(r)
        global vxlist, edge_list = [i for i in 1:n_poiss], Base.copy(edges)
        global independent_edges = find_independent_edges(vxlist,edge_list,p0)
        display(independent_edges)
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
plt = plot([], []; xlims=(-sqrt(num_points)/1.95,sqrt(num_points)/1.95), ylims=(-sqrt(num_points)/1.95,sqrt(num_points)/1.95), linewidth=4, color=:steelblue, grid=false, label="", size=(800,800), tickfontsize=16)
for edge in edges
    plot!(plt, [p0[1,edge[1]], p0[1,edge[2]]], [p0[2,edge[1]], p0[2,edge[2]]], arrow=false, color=:steelblue, linewidth=4, label="")
end
for i in 1:size(p0)[2]
    scatter!(plt, [p0[1,i]], [p0[2,i]], color=:black, mc=:black, markersize=7, label="")
end
savefig(plt, "GilbertGraphTest.png")

barequations = [sum((xs[:,bar[1]]-xs[:,bar[2]]).^2) - sum((p0[:,bar[1]]-p0[:,bar[2]]).^2) for bar in edges]
dg = nullspace(evaluate(differentiate(barequations, xvarz), xvarz=>p))
#barequations = rand(Float64,num_points*2-4,length(barequations))*barequations
v = 1*Vector{Float64}(dg[:,1])
G = Euclidean_distance_retraction_minimal.ConstraintVariety(xvarz,barequations,num_points*2-3,1)
#=F = System(barequations, variables=xvarz)
display(F)
Euclidean_distance_retraction_minimal.HCnewtonstep(F,p)=#
EDStep = (i,euler_step)->Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="gaussnewton", euler_step=euler_step, amount_Euler_steps=i)
#display(rank(evaluate(G.EDTracker.jacobian, G.EDTracker.tracker.homotopy.F.interpreted.system.variables=>vcat(p,0))))
euler_array = ["newton", "explicit"]

testDict["GilbertGraph"] = Dict()
for euler in 1:length(euler_array)
    testDict["GilbertGraph"][euler_array[euler]] = []
    global index = 0
    global method = euler_array[euler]
    while index <= max_indices
        euler_array[euler]=="newton" ? println("$(index) Newton Discretization Steps") : println("$(index) nontrivial Euler Steps")
        local u = median(@benchmark EDStep(index,method))
        
        #println("Average Linear Steps: ", sum(linear_steps)/length(linear_steps))
        #println("Solution: ", EDStep(index,method)[1])
        global index = index+1
        push!(testDict["GilbertGraph"][euler_array[euler]], (u.time/(1000*1000), u.memory/1000, EDStep(index,method)[2]))
        q = evaluate.(xs, xvarz=>EDStep(index,method)[1])
        for edge in edges
            plot!(plt, [q[1,edge[1]], q[1,edge[2]]], [q[2,edge[1]], q[2,edge[2]]], arrow=false, color=:lightgrey, linewidth=4, label="")
        end
        for i in 1:size(q)[2]
            if !isapprox(norm(p0[:,i]-q[:,i]), 0)
                plot!(plt, [p0[1,i], q[1,i]], [p0[2,i], q[2,i]], arrow=true, color=:green3, linewidth=4, label="")
            end
        end
        for edge in edges
            plot!(plt, [p0[1,edge[1]], p0[1,edge[2]]], [p0[2,edge[1]], p0[2,edge[2]]], arrow=false, color=:steelblue, linewidth=4, label="")
        end
        for i in 1:size(p0)[2]
            scatter!(plt, [p0[1,i]], [p0[2,i]], color=:black, mc=:black, markersize=7, label="")
        end
        savefig(plt, "GilbertGraphTest2.png")
    end
end
println("HC.jl")
u = median(@benchmark Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
println("Solution: ", Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation"))
linear_steps = Euclidean_distance_retraction_minimal.EDStep(G, p, v; homotopyMethod="HomotopyContinuation")[2]
println("Average Linear Steps: ", )
testDict["GilbertGraph"]["HC.jl"] = [(u.time/(1000*1000), u.memory/1000, sum(linear_steps)/length(linear_steps))]
display(testDict)






open("testEDResults.txt", "w") do file
    for key in keys(testDict)
        write(file, "$(key)\n")
        for key2 in keys(testDict[key])
            write(file, "$(key2)\n")
            for i in 1:length(testDict[key][key2])
                write(file, "$(i): $(testDict[key][key2][i][1])ms, $(testDict[key][key2][i][2])kb, $(testDict[key][key2][i][3]) linear solves\n")
            end
        end
        write(file, "\n")
    end
end
