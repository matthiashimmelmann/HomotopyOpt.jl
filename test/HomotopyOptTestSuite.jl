include("../src/HomotopyOpt_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random

Random.seed!(1234);
N_samples=300

#=
#=
G = HomotopyOpt.ConstraintVariety(x->[x[1]^2+x[2]^2-1],2,1,N_samples)
objective = x->(x[1]-2)^2+x[2]^2
display(HomotopyOpt.findminima([0,1.], 1e-3, G, objective; maxseconds=50, whichstep="Algorithm 2"))

sleep(30)=#
#INFO: You can set the amount of samples here:
#filename = open("testSuiteResults.txt", "w");
@var y[1:3] l
R1,R2=2,1
L = (y[1]-2)^2+y[2]^2+(y[3]-2)^2 + l*((y[1]^2+y[2]^2+y[3]^2-R1^2-R2^2)^2/(4*R1^2)+y[3]^2-R2^2)
dL = HomotopyContinuation.differentiate(L, vcat(y,l))
sols=HomotopyContinuation.real_solutions(HomotopyContinuation.solve(dL))

f1 = x->[(x[1]^2+x[2]^2+x[3]^2-R1^2-R2^2)^2/(4*R1^2)+x[3]^2-R2^2]
G = HomotopyOpt.ConstraintVariety(f1,3,2,N_samples)
norm = t->sqrt(t[1]^2+t[2]^2+t[3]^2)
G.samples = filter(t->norm(t)<1000, G.samples)
G.samples = length(G.samples)>N_samples ? G.samples[1:N_samples] : G.samples
objective = x->(x[1]-2)^2+x[2]^2+(x[3]-2)^2
HomotopyOpt.findminima(G.samples[1], 1e-3, G, objective; maxseconds=100, whichstep="HomotopyContinuation")
println("Torus Test")

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time3 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=100, whichstep="Algorithm 0")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 0...\t\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time3 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=100, whichstep="Algorithm 0.1")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 0+Euler...\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time1 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=100, whichstep="Algorithm 1")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 1...\t\t", "avg. time: ", round(1000*(Base.time()-time1)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time2 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=100, whichstep="Algorithm 2")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 2...\t\t", "avg. time: ", round(1000*(Base.time()-time2)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end


let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time4 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=100, whichstep="HomotopyContinuation")
        global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("HomotopyContinuation...\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    println(" ")
end
=#

f1 = x->[x[1]^2-x[2], x[1]^3-x[3]]
G = HomotopyOpt.ConstraintVariety(f1,3,1,N_samples)
norm = t->sqrt(t[1]^2+t[2]^2+t[3]^2)
G.samples = filter(t->norm(t)<100, G.samples)
G.samples = length(G.samples)>N_samples ? G.samples[1:N_samples] : G.samples
objective = x->x[1]^2+(x[2]+1)^2+(x[3]-1)^2
#display(HomotopyOpt.findminima([1.,1,1], 1e-3, G, objective; maxseconds=100, whichstep="HomotopyContinuation").converged)
#HomotopyOpt.findminima([1.,-1,1], 1e-3, G, objective; homotopyMethod = "HomotopyContinuation")
println("Twisted Cubic Test")

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time3 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=10, whichstep="Algorithm 0")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 0...\t\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time3 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=10, whichstep="Algorithm 0.1")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 0+Euler...\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

#=
let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time1 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=10, whichstep="Algorithm 1")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 1...\t\t", "avg. time: ", round(1000*(Base.time()-time1)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time2 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=10, whichstep="Algorithm 2")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 2...\t\t", "avg. time: ", round(1000*(Base.time()-time2)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end
=#
let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time4 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=100, whichstep="HomotopyContinuation")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("HomotopyContinuation...\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    println(" ")
end




@var y[1:2] l
L = (y[1]-0.5)^2+(y[2]-2)^2 + l*((y[1]^4+y[2]^4-1)*(y[1]^2+y[2]^2-2)+y[1]^5*y[2])
dL = HomotopyContinuation.differentiate(L, vcat(y,l))
sols=HomotopyContinuation.real_solutions(HomotopyContinuation.solve(dL))
#display(sols)

f1 = x->[((x[1]^4+x[2]^4-1)*(x[1]^2+x[2]^2-2)+x[1]^5*x[2])]
G = HomotopyOpt.ConstraintVariety(f1,2,1,N_samples)
norm = t->sqrt(t[1]^2+t[2]^2)
G.samples = filter(t->norm(t)<100, G.samples)
G.samples = length(G.samples)>N_samples ? G.samples[1:N_samples] : G.samples
objective = x->(x[1]-0.5)^2+(x[2]-2)^2
HomotopyOpt.findminima([0.,1], 1e-3, G, objective; maxseconds=100, whichstep="HomotopyContinuation")
println("Planar Sextic Test")

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time3 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1,whichstep="Algorithm 0")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 0...\t\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time3 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1,whichstep="Algorithm 0.1")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 0+Euler...\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

#=
let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time1 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective;  maxseconds=10, maxlocalsteps=1, whichstep="Algorithm 1")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 1...\t\t", "avg. time: ", round(1000*(Base.time()-time1)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time2 = Base.time()
    for pt in G.samples
        local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=10, maxlocalsteps=1, whichstep="Algorithm 2")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("Algorithm 2...\t\t", "avg. time: ", round(1000*(Base.time()-time2)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end
=#

let
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time4 = Base.time()
    for pt in G.samples
        res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=20, maxlocalsteps=1, whichstep="HomotopyContinuation")
        global convergedPaths = convergedPaths+res.converged
        global localSteps = localSteps + (length(res.computedpoints)-1)
        global isMinimum = isMinimum + res.lastpointisminimum
    end
    println("HomotopyContinuation...\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    println(" ")
end




function gaussnewtonstep(G, p, stepsize, v; tol=1e-8)
	global q = p+stepsize*v
	global damping = 0.5
	global qnew = q
	jac = hcat([differentiate(eq, G.variables) for eq in G.fullequations]...)
	while(norm(evaluate.(G.fullequations, G.variables=>q)) > tol)
		J = Matrix{Float64}(evaluate.(jac, G.variables=>q))
		global qnew = q .- damping*pinv(J)'*evaluate.(G.fullequations, G.variables=>q)
		if norm(evaluate.(G.fullequations, G.variables=>qnew)) <= norm(evaluate.(G.fullequations, G.variables=>q))
			global damping = damping*1.2
		else
			global damping = damping/2
		end
		q = qnew
	end
	return q
end

println("Stiefel Manifold Test")
for n in [(2,2), (3,3), (4,4), (5,5)]
    println("$(n)...")
    @var x[1:n[1],1:n[2]]
    f3 = vcat(x*x'-LinearAlgebra.Diagonal([1 for _ in 1:n[1]])...)
    xvarz = vcat([x[i,j] for i in 1:n[1], j in 1:n[2]]...)
    global G = HomotopyOpt.ConstraintVariety(xvarz, f3, n[1]*n[2], n[1]*n[2]-Int(n[1]*(n[1]+1)/2), n[2]>=4 ? 0 : Int(N_samples/(n[2]+1)))
    G.samples = n[2]>=4 ? filter(t -> norm(t)<250, [gaussnewtonstep(G, 10 .* (rand(Float64,n[1]*n[2]) .- 0.5), 0, [0 for _ in 1:n[1]*n[2]]; tol=5e-15) for _ in 1:N_samples]) : filter(t->norm(t)<250, G.samples)
    G.samples = filter(t->norm(t)<500, G.samples)
    G.samples = length(G.samples)>N_samples ? G.samples[1:N_samples] : G.samples
    obj = rand(Float64,n[1]*n[2])
    global objective = x->sum((x[i] - 1)^2 for i in 1:n[1]*n[2])
    HomotopyOpt.findminima(gaussnewtonstep(G, obj, 0, [0 for _ in 1:n[1]*n[2]]; tol=5e-15), 1e-3, G, objective; maxlocalsteps=1, maxseconds=150, whichstep="HomotopyContinuation")

    let 
        global convergedPaths, localSteps, isMinimum = 0, 0, 0
        time3 = Base.time()
        for pt in G.samples
            local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=250, maxlocalsteps=1,whichstep="Algorithm 0")
            global convergedPaths = convergedPaths+res.converged
            global localSteps = localSteps + (length(res.computedpoints)-1)
            global isMinimum = isMinimum + res.lastpointisminimum
        end
        println("Algorithm 0...\t\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    end
    
    let 
        global convergedPaths, localSteps, isMinimum = 0, 0, 0
        time3 = Base.time()
        for pt in G.samples
            local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=250, maxlocalsteps=1,whichstep="Algorithm 0.1")
            global convergedPaths = convergedPaths+res.converged
            global localSteps = localSteps + (length(res.computedpoints)-1)
            global isMinimum = isMinimum + res.lastpointisminimum
        end
        println("Algorithm 0+Euler...\t", "avg. time: ", round(1000*(Base.time()-time3)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    end
    #=
    let
        global convergedPaths, localSteps, isMinimum = 0, 0, 0
        time1 = Base.time()
        for pt in G.samples
            local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=250, maxlocalsteps=1, whichstep="Algorithm 1")
            global convergedPaths = convergedPaths+res.converged
            global localSteps = localSteps + (length(res.computedpoints)-1)
            global isMinimum = isMinimum + res.lastpointisminimum
        end
        println("Algorithm 1...\t\t", "avg. time: ", round(1000*(Base.time()-time1)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    end
    
    let
        global convergedPaths, localSteps, isMinimum = 0, 0, 0
        time2 = Base.time()
        for pt in G.samples
            local res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxseconds=250, maxlocalsteps=1, whichstep="Algorithm 2")
            global convergedPaths = convergedPaths+res.converged
            global localSteps = localSteps + (length(res.computedpoints)-1)
            global isMinimum = isMinimum + res.lastpointisminimum
        end
        println("Algorithm 2...\t\t", "avg. time: ", round(1000*(Base.time()-time2)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
    end
    =#
    let
        global convergedPaths, localSteps, isMinimum = 0, 0, 0
        time4 = Base.time()
        for pt in G.samples
            res = HomotopyOpt.findminima(pt, 1e-3, G, objective; maxlocalsteps=1, maxseconds=250, whichstep="HomotopyContinuation")
            global convergedPaths = res.converged ? convergedPaths+1 : convergedPaths
            global localSteps = localSteps + (length(res.computedpoints)-1)
            global isMinimum = isMinimum + res.lastpointisminimum
        end
        println("HomotopyContinuation\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms,\t% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
        println(" ")
    end
end
#close(filename)