include("../src/HomotopyOpt_minimal.jl")
using HomotopyContinuation, LinearAlgebra, Random

Random.seed!(1234);
N_samples=300
#=
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

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time4 = Base.time()
    println("Lagrange Multiplier...\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time4 = Base.time()
    println("ManOpt...\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
end

let 
    global convergedPaths, localSteps, isMinimum = 0, 0, 0
    time4 = Base.time()
    println("Inner Point...\t", "avg. time: ", round(1000*(Base.time()-time4)/length(G.samples), digits=1), "ms,\t", "% converged: ", round(100*convergedPaths/length(G.samples), digits=1), ",\tavg. local steps: ", round(localSteps/length(G.samples), digits=1), ",\t% minimal: ", round(100*convergedPaths/length(G.samples), digits=1))
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
=#
function gaussnewtonstep(G, p; tol=1e-8)
	global damping = 0.5
	global qnew,q = p,p
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

function toArray(p)
    configuration = Vector{Float64}([])
    for i in 1:3, j in 4:6
        push!(configuration, p[i,j])
    end
    return configuration
end

function toMatrix(p,p0)
    output = Base.copy(p0)
    count = 1
    for i in 1:3, j in 4:6
        output[i,j] = p[count]
        count +=1
    end
    return output
end


println("Bricard Octahedron Test")
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
global v = real.(nlp[:,1] ./ (norm(nlp[:,1])*nlp[1,1]))
global n = vcat(cross(v[1:3],[1,1,1]),cross(v[4:6],[1,1,1]),cross(v[7:9],[1,1,1]))
global cursol = toArray(p0)
solutionarray1, solutionarray2, solutioncurve = [],[],[]
@var u[1:length(v)] λ[1:length(barequations)]
L = sum((xvarz .- u).^2) + λ'*barequations
dL = differentiate(L, vcat(xvarz, λ))
dgL = differentiate(dL, vcat(xvarz, λ))
λ0 = randn(Float64, length(λ))
global cursol = vcat(toArray(p0), λ0)
while norm(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0))))>1e-12
    global cursol = cursol - pinv(real.(evaluate(dgL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0)))))*real.(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0))))
end


for t in 0.01:0.01:2.5
    q = toArray(p0)+t*v
    global cursol = cursol .- pinv(real.(evaluate.(dgL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0)+(t-0.01)*v)))) * real.(evaluate.(differentiate(dL,u), vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0)+(t-0.01)*v))) * (0.05*v)
    while norm(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,q)))>1e-10
        global cursol = cursol - pinv(real.(evaluate(dgL, vcat(xvarz,λ,u)=>vcat(cursol,q))))*real.(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,q)))
    end
    push!(solutionarray1, cursol[1:length(xvarz)])
end

global cursol = vcat(toArray(p0), λ0)
while norm(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0))))>1e-12
    global cursol = cursol - pinv(real.(evaluate(dgL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0)))))*real.(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0))))
end
for t in 0.01:0.01:2.5
    q = toArray(p0)-t*v
    global cursol = cursol .- pinv(real.(evaluate.(dgL, vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0)+(t-0.01)*v)))) * real.(evaluate.(differentiate(dL,u), vcat(xvarz,λ,u)=>vcat(cursol,toArray(p0)+(t-0.01)*v))) * (0.05*v)
    while norm(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,q)))>1e-10
        global cursol = cursol - pinv(real.(evaluate(dgL, vcat(xvarz,λ,u)=>vcat(cursol,q))))*real.(evaluate(dL, vcat(xvarz,λ,u)=>vcat(cursol,q)))
    end
    push!(solutionarray2, cursol[1:length(xvarz)])
end

for el in vcat(solutionarray2[end:-1:1], solutionarray1)
    push!(solutioncurve, toMatrix(el[1:length(xvarz)], p0))
end

triangs = [ (1,2,3), (6,3,2), (6,4,5),  (1,3,4), (6,4,3), (1,2,5), (6,5,2), (1,4,5)]
for index in 1:length(solutioncurve)
    open("BricardOctahedron/deformation$(index).poly", "w") do f
        write(f, "POINTS\n")
        foreach(i->write(f, string("$(i): ", solutioncurve[index][1,i], " ", solutioncurve[index][2,i], " ", solutioncurve[index][3,i],"\n")), 1:size(solutioncurve[index])[2])
        write(f,"POLYS\n")
        for i in 1:length(triangs)
            write(f, string("$(i): "))
            for j in 1:length(triangs[i])-1
                write(f, "$(triangs[i][j]) ")
            end
            write(f, "$(triangs[i][end]) <\n")
        end
        write(f,"END")
    end
end

