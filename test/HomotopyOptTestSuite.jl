include("./HomotopyOpt.jl/src/HomotopyOpt.jl")
using HomotopyContinuation

println("Twisted Cubic Test")
@var y[1:3] l[1:2]
L = y[1]^2+(y[2]+1)^2+(y[3]-1)^2 + l'*[y[1]^2-y[2], y[1]^3-y[3]]
dL = HomotopyContinuation.differentiate(L, vcat(y,l))
sols=HomotopyContinuation.real_solutions(HomotopyContinuation.solve(dL))
display(sols)

f1 = x->[x[1]^2-x[2], x[1]^3-x[3]]
G = HomotopyOpt.ConstraintVariety(f1,3,1,100)
norm = t->sqrt(t[1]^2+t[2]^2+t[3]^2)
G.samples = filter(t->norm(t)<1000, G.samples)
objective = x->x[1]^2+(x[2]+1)^2+(x[3]-1)^2
abc = HomotopyOpt.findminima([1.,-1,1], 1e-3, G, objective; homotopyMethod = "Newton")
#HomotopyOpt.findminima([1.,-1,1], 1e-3, G, objective; homotopyMethod = "HomotopyContinuation")

global convergedPaths = 0
global localSteps = 0
time1 = Base.time()
for pt in G.samples
    res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, homotopyMethod = "Newton")
    global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
    global localSteps = localSteps + length(res.computedpoints)
end
println("Newton... ", "Average time: ",(Base.time()-time1)/length(G.samples), "s, ", "% converged: ", 100*convergedPaths/length(G.samples), ", Average local steps: ", localSteps/length(G.samples))

global convergedPaths = 0
global localSteps = 0
time2 = Base.time()
for pt in G.samples
    res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, homotopyMethod = "HomotopyContinuation")
    global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
    global localSteps = localSteps + length(res.computedpoints)
end
println("HomotopyContinuation... ", "Average time: ",(Base.time()-time2)/length(G.samples), "s, ", "% converged: ", 100*convergedPaths/length(G.samples), ", Average local steps: ", localSteps/length(G.samples))
println(" ")


println("Planar Sextic Test")

@var y[1:2] l
L = (y[1]-0.5)^2+(y[2]-2)^2 + l*((y[1]^4+y[2]^4-1)*(y[1]^2+y[2]^2-2)+y[1]^5*y[2])
dL = HomotopyContinuation.differentiate(L, vcat(y,l))
sols=HomotopyContinuation.real_solutions(HomotopyContinuation.solve(dL))
#display(sols)

f1 = x->[((x[1]^4+x[2]^4-1)*(x[1]^2+x[2]^2-2)+x[1]^5*x[2])]
G = HomotopyOpt.ConstraintVariety(f1,2,1,100)
norm = t->sqrt(t[1]^2+t[2]^2)
G.samples = filter(t->norm(t)<1000, G.samples)
objective = x->(x[1]-0.5)^2+(x[2]-2)^2
abc = HomotopyOpt.findminima([0.,1], 1e-3, G, objective; homotopyMethod = "Newton")
#HomotopyOpt.findminima([1.,-1,1], 1e-3, G, objective; homotopyMethod = "HomotopyContinuation")

global convergedPaths = 0
global localSteps = 0
time1 = Base.time()
for pt in G.samples
    res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, homotopyMethod = "Newton")
    global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
    global localSteps = localSteps + length(res.computedpoints)
end
println("Newton... ", "Average time: ",(Base.time()-time1)/length(G.samples), "s, ", "% converged: ", 100*convergedPaths/length(G.samples), ", Average local steps: ", localSteps/length(G.samples))

global convergedPaths = 0
global localSteps = 0
time2 = Base.time()
for pt in G.samples
    res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, homotopyMethod = "HomotopyContinuation")
    global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
    global localSteps = localSteps + length(res.computedpoints)
end
println("HomotopyContinuation... ", "Average time: ",(Base.time()-time2)/length(G.samples), "s, ", "% converged: ", 100*convergedPaths/length(G.samples), ", Average local steps: ", localSteps/length(G.samples))
println(" ")



println("Torus Test")
@var y[1:3] l
R1,R2=2,1
L = (y[1]-2)^2+y[2]^2+(y[3]-2)^2 + l*((y[1]^2+y[2]^2+y[3]^2-R1^2-R2^2)^2/(4*R1^2)+y[3]^2-R2^2)
dL = HomotopyContinuation.differentiate(L, vcat(y,l))
sols=HomotopyContinuation.real_solutions(HomotopyContinuation.solve(dL))

f1 = x->[(x[1]^2+x[2]^2+x[3]^2-R1^2-R2^2)^2/(4*R1^2)+x[3]^2-R2^2]
G = HomotopyOpt.ConstraintVariety(f1,3,2,100)
norm = t->sqrt(t[1]^2+t[2]^2+t[3]^2)
G.samples = filter(t->norm(t)<1000, G.samples)
objective = x->(x[1]-2)^2+x[2]^2+(x[3]-2)^2
abc = HomotopyOpt.findminima([-2.,0,1], 1e-3, G, objective; homotopyMethod = "Newton")

global convergedPaths = 0
global localSteps = 0
time1 = Base.time()
for pt in G.samples
    res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, homotopyMethod = "Newton")
    global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
    global localSteps = localSteps + length(res.computedpoints)
end
println("Newton... ", "Average time: ",(Base.time()-time1)/length(G.samples), "s, ", "% converged: ", 100*convergedPaths/length(G.samples), ", Average local steps: ", localSteps/length(G.samples))

global convergedPaths = 0
global localSteps = 0
time2 = Base.time()
for pt in G.samples
    res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, homotopyMethod = "HomotopyContinuation")
    global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
    global localSteps = localSteps + length(res.computedpoints)
end
println("HomotopyContinuation... ", "Average time: ",(Base.time()-time2)/length(G.samples), "s, ", "% converged: ", 100*convergedPaths/length(G.samples), ", Average local steps: ", localSteps/length(G.samples))
println(" ")
