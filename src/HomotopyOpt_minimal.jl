module HomotopyOpt

import HomotopyContinuation: @var, evaluate, differentiate, start_parameters!, target_parameters!, track!, solve, real_solutions, solutions, solution, rand_subspace, randn, System, ParameterHomotopy, Expression, Tracker, Variable, track
import LinearAlgebra: norm, transpose, qr, rank, normalize, pinv, eigvals, abs, eigvecs, svd, nullspace
import Plots: plot, scatter!, Animation, frame
import ForwardDiff: hessian, gradient
import HomotopyContinuation

export ConstraintVariety,
       findminima,
       watch,
       draw,
	   addSamples!,
	   setEquationsAtp!,
#INFO: The following package is not maintained by us. Find it here: https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl
	   HomotopyContinuation

#=
 Equips a HomotopyContinuation.Tracker with a start Solution that can be changed on the fly
=#
mutable struct TrackerWithStartSolution
	tracker
	startSolution
    jacobian
    ptv
    jacobian_parameter
	#basepoint

	function TrackerWithStartSolution(T::Tracker, startSol::Vector, d::Int)
        @var t point[1:d] vector[1:d]
        jacobian_z = hcat([differentiate(eq, T.homotopy.F.interpreted.system.variables) for eq in T.homotopy.F.interpreted.system.expressions]...)
		jacobian_t = [differentiate(eq, t) for eq in evaluate(T.homotopy.F.interpreted.system.expressions, T.homotopy.F.interpreted.system.parameters=>point .+ t .* vector)]
        new(T, startSol, jacobian_z, Vector{Variable}(vcat(t,point,vector)), jacobian_t)
	end
end

function setStartSolution(T::TrackerWithStartSolution, startSol::Vector)
	setfield!(T, :startSolution, startSol)
end

#=
 An object that describes a constraint variety by giving its generating equations, coordinate variables, its dimension and its jacobian.
 Additionally, it contains the system describing the Euclidian Distance Problem and samples from the variety.
=#
mutable struct ConstraintVariety
    variables
    equations
	fullequations
    jacobian
    ambientdimension
	dimensionofvariety
    samples
    implicitequations
    EDTracker

	# Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
	function ConstraintVariety(varz, eqnz, N::Int, d::Int, numsamples::Int)
        jacobian = hcat([differentiate(eq, varz) for eq in eqnz]...)
		impliciteq = [p->eqn(varz=>p) for eqn in eqnz]
        randL = nothing
		randresult = nothing
        Ωs = []
		if numsamples > 0
			randL = rand_subspace(N; codim=d)
			randResult = solve(eqnz; target_subspace = randL, variables=varz, show_progress = true)
		end
        for _ in 1:numsamples
            newΩs = solve(
                    eqnz,
                    solutions(randResult);
                    variables = varz,
                    start_subspace = randL,
                    target_subspace = rand_subspace(N; codim = d, real = true),
                    transform_result = (R,p) -> real_solutions(R),
                    flatten = true,
					show_progress = true
            )
            realsols = real_solutions(newΩs)
            push!(Ωs, realsols...)
        end
		Ωs = filter(t -> norm(t)<1e4,Ωs)
		fulleqnz = eqnz
		if length(eqnz) + d > N
			eqnz = randn(Float64, N-d, length(eqnz))*eqnz
		end

        @var u[1:N]
		@var λ[1:length(eqnz)]
		Lagrange = sum((varz-u).^2) + sum(λ.*eqnz)
		∇Lagrange = differentiate(Lagrange, vcat(varz,λ))
		EDSystem = System(∇Lagrange, variables=vcat(varz,λ), parameters=u)
		p0 = randn(Float64, N)
		H = ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
		EDTracker = TrackerWithStartSolution(Tracker(H),[], N)
        new(varz,eqnz,fulleqnz,jacobian,N,d,Ωs,impliciteq,EDTracker)
    end

	# Given implicit equations, sample points from the corresponding variety and return the struct
    function ConstraintVariety(eqnz::Function, N::Int, d::Int, numsamples::Int)
        @var varz[1:N]
        algeqnz = eqnz(varz)
		if typeof(algeqnz) != Vector{Expression}
			algeqnz = [algeqnz]
		end
		ConstraintVariety(varz, algeqnz, N::Int, d::Int, numsamples::Int)
    end

	# Implicit Equations, no sampling
    function ConstraintVariety(eqnz,N::Int,d::Int)
		ConstraintVariety(eqnz::Function, N::Int, d::Int, 0)
	end

    # HomotopyContinuation-based expressions and variables, no sanples
    function ConstraintVariety(varz,eqnz,N::Int,d::Int)
		ConstraintVariety(varz, eqnz, N::Int, d::Int, 0)
    end

	#Let the dimension be determined by the algorithm and calculate samples
	function ConstraintVariety(varz,eqnz,p::Vector{Float64},numSamples::Int)
		G = ConstraintVariety(varz, eqnz, length(varz), 0,numSamples)
		setEquationsAtp!(G,p)
		return(G)
	end

	#Only let the dimension be determined by the algorithm
	function ConstraintVariety(varz,eqnz,p::Vector{Float64})
		G = ConstraintVariety(varz, eqnz, length(varz), 0)
		setEquationsAtp!(G,p)
		return(G)
	end
end

#=
Add Samples to an already existing ConstraintVariety
=#
function addSamples!(G::ConstraintVariety, newSamples)
	setfield!(G, :samples, vcat(newSamples, G.samples))
end

#=
Add Samples to an already existing ConstraintVariety
=#
function setEquationsAtp!(G::ConstraintVariety, p; tol=1e-5)
	jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol=tol)
	eqnz = G.fullequations
	if length(eqnz) + (G.ambientdimension-jacobianRank) > G.ambientdimension
		eqnz = randn(Float64, jacobianRank, length(eqnz))*eqnz
	end
	setfield!(G, :equations, eqnz)
	setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))
end


#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep(G::ConstraintVariety, p; tol=1e-8)
	global q = p
	while(norm(evaluate.(G.fullequations, G.variables=>q)) > tol)
		J = Matrix{Float64}(evaluate.(G.jacobian, G.variables=>q))
		global q = q .- pinv(J)'*evaluate.(G.fullequations, G.variables=>q)
	end
	return q
end

#=
 Checks, whether p is a local minimum of the objective function Q w.r.t. the tangent space Tp
=#
function isMinimum(G::ConstraintVariety, Q::Function, evaluateobjectivefunctiongradient, Tp, v, p::Vector; tol=1e-4, criticaltol=1e-3)
	H = hessian(Q, p)
	HConstraints = [evaluate.(differentiate(differentiate(eq, G.variables), G.variables), G.variables=>p) for eq in G.fullequations]
	Qalg = Q(p)+(G.variables-p)'*gradient(Q,p)+0.5*(G.variables-p)'*H*(G.variables-p) # Taylor Approximation of x, since only the Hessian is of interest anyway
	@var λ[1:length(G.fullequations)]
	L = Qalg+λ'*G.fullequations
	∇L = differentiate(L, vcat(G.variables, λ))
	gL = Matrix{Float64}(evaluate(differentiate(∇L, λ), G.variables=>p))
	bL = -evaluate.(evaluate(∇L,G.variables=>p), λ=>[0 for _ in 1:length(λ)])
	λ0 = map( t-> (t==NaN || t==Inf) ? 1 : t, gL\bL)

	Htotal = H+λ0'*HConstraints
	projH = Matrix{Float64}(Tp'*Htotal*Tp)
	projEigvals = real(eigvals(projH)) #projH symmetric => all real eigenvalues
	#println("Eigenvalues of the projected Hessian: ", round.(1000 .* projEigvals, sigdigits=3) ./ 1000)
	indices = filter(i->abs(projEigvals[i])<=tol, 1:length(projEigvals))
	projEigvecs = real(eigvecs(projH))[:, indices]
	projEigvecs = Tp*projEigvecs
	if all(q-> q>=tol, projEigvals) && norm(v) <= criticaltol
		return true
	elseif any(q-> q<=-tol, projEigvals) || norm(v) > criticaltol
		return false
		#TODO Third derivative at x_0 at proj hessian sing. vectors not 0?!
	# Else take a small step in gradient descent direction and see if the energy decreases
	else
		q = gaussnewtonstep(G, p-1e-2*evaluateobjectivefunctiongradient(p)[2])
		return Q(q)<Q(p)
	end
end

function EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod, tol=1e-10, disc=0.1, initialtime, maxseconds)
	if homotopyMethod=="HomotopyContinuation"
        q = p+stepsize*v
		target_parameters!(ConstraintVariety.EDTracker.tracker, q)
		tracker = track(ConstraintVariety.EDTracker.tracker, ConstraintVariety.EDTracker.startSolution)
		result = solution(tracker)
		if all(point->Base.abs(point.im)<1e-4, result)
			return [point.re for point in result[1:length(p)]]
		else
			throw(error("Complex Space entered!"))
		end
    elseif homotopyMethod=="Algorithm -1"
		return gaussnewtonstep(ConstraintVariety, p+stepsize*v; tol=tol)
	elseif homotopyMethod=="Algorithm 0"
        q = p+stepsize*v
		currentSolution = vcat(q, ConstraintVariety.EDTracker.startSolution[length(q)+1:end])
		variables = ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables
		equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
        jac = Matrix{Expression}(evaluate.(ConstraintVariety.EDTracker.jacobian, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q))
        while(norm(evaluate.(equations, variables=>currentSolution)) > tol)
            if Base.time()-initialtime > maxseconds
                break
            end
			J = evaluate.(jac, variables=>currentSolution)
			currentSolution =  currentSolution .- J \ evaluate.(equations, variables=>currentSolution)
		end
		return currentSolution[1:length(q)]
    elseif homotopyMethod=="Algorithm 0.1"
        q = p+stepsize*v
		currentSolution = vcat(p, ConstraintVariety.EDTracker.startSolution[length(p)+1:end])
		variables = ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables
		equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
        jac = Matrix{Expression}(evaluate.(ConstraintVariety.EDTracker.jacobian, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q))
        #println(currentSolution, stepsize*v, EulerStep(ConstraintVariety, u_t, t, p, 1, currentSolution))
        currentSolution = currentSolution .+ EulerStep(ConstraintVariety, currentSolution, p, stepsize*v, 0, 1)
        while(norm(evaluate.(equations, variables=>currentSolution)) > tol)
            if Base.time()-initialtime > maxseconds
                break
            end
			J = evaluate.(jac, variables=>currentSolution)
			currentSolution =  currentSolution .- J \ evaluate.(equations, variables=>currentSolution)
		end
        #println(norm(prev_sol-currentSolution), " ", norm(prediction1-currentSolution), " ", stepsize)
		return currentSolution[1:length(q)]
    elseif homotopyMethod=="Algorithm 1"
        curL = ConstraintVariety.EDTracker.startSolution[length(p)+1:end]
        q = p+disc*stepsize*v
        currentSolution = vcat(p, curL)
        variables = ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables

        for t in disc:disc:1
            q = p+t*stepsize*v
            equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
            while(norm(evaluate.(equations, variables=>currentSolution)) > tol)
                if Base.time()-initialtime > maxseconds
                    break
                end
                J = evaluate.(ConstraintVariety.EDTracker.jacobian, vcat(variables, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(currentSolution, q))
                currentSolution =  currentSolution .- J \ evaluate.(equations, variables=>currentSolution)
            end
            curL = currentSolution[length(p)+1:end]
        end
        return currentSolution[1:length(q)]    
    elseif homotopyMethod=="Algorithm 2"
        curL = ConstraintVariety.EDTracker.startSolution[length(p)+1:end]
        q = p+disc*stepsize*v
        currentSolution = vcat(p, curL)
        variables = ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables

        for step in disc:disc:1
            q = p+step*stepsize*v
            prev_sol = currentSolution
            currentSolution = currentSolution .+ EulerStep(ConstraintVariety, currentSolution, p, stepsize*v, step-disc, disc)
            equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
            while(norm(evaluate.(equations, variables=>currentSolution)) > tol)
                if Base.time()-initialtime > maxseconds
                    break
                end
                J = evaluate.(ConstraintVariety.EDTracker.jacobian, vcat(variables, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(currentSolution, q))
                currentSolution =  currentSolution .- J \ evaluate.(equations, variables=>currentSolution)
            end
            curL = currentSolution[length(p)+1:end]
        end
        #println(norm(prev_sol-currentSolution), " ", norm(prediction-currentSolution))
        return currentSolution[1:length(q)]
	else
		throw(error("Homotopy Method not supported!"))
	end
end

function EulerStep(ConstraintVariety, q, p, v, prev_step, step_size)
    dz = -evaluate.(ConstraintVariety.EDTracker.jacobian, vcat(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(q, p+prev_step*v))
    du = evaluate.(ConstraintVariety.EDTracker.jacobian_parameter, vcat(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables, ConstraintVariety.EDTracker.ptv) => vcat(q, p, prev_step, v))
    return dz \ (du*step_size)
end

#=
Determines, which optimization algorithm to use
=#
function stepchoice(ConstraintVariety, whichstep, stepsize, p, v; initialtime=initialtime, maxseconds=maxseconds)
	if whichstep=="Algorithm 0"
        q = EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod="Algorithm 0", initialtime=initialtime, maxseconds=maxseconds)
		return q
    elseif whichstep=="Algorithm 0.1"
        q = EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod="Algorithm 0.1", initialtime=initialtime, maxseconds=maxseconds)
        return q    
	elseif whichstep=="Algorithm 1"
        q = EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod="Algorithm 1", initialtime=initialtime, maxseconds=maxseconds)
        return q
    elseif whichstep=="Algorithm 2"
        q = EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod="Algorithm 2", initialtime=initialtime, maxseconds=maxseconds)
        return q
    elseif whichstep=="HomotopyContinuation"
        q = EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod="HomotopyContinuation", initialtime=initialtime, maxseconds=maxseconds)
        return q
    else
		throw(error("A step method needs to be provided!"))
	end
end

#=
Use line search with the strong Wolfe condition to find the optimal step length.
This particular method can be found in Nocedal & Wright: Numerical Optimization
=#
function backtracking_linesearch(Q::Function, G::ConstraintVariety, evaluateobjectivefunctiongradient::Function, p0::Vector, stepsize::Float64; whichstep="EDStep", r=1e-4, s=0.9, maxstepsize, initialtime=initialtime, maxseconds=maxseconds)
	Basenormal, _, basegradient, _ = get_NTv(p0, G, evaluateobjectivefunctiongradient)
	α = [0, stepsize]
	p = Base.copy(p0)

    q0 = p+1e-3*Basenormal[:,1]
    start_parameters!(G.EDTracker.tracker, q0)
    A = evaluate.(differentiate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end]), G.variables => p)
    λ0 = A\-evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, vcat(G.EDTracker.tracker.homotopy.F.interpreted.system.variables, G.EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(p, [0 for _ in length(p)+1:length(G.EDTracker.tracker.homotopy.F.interpreted.system.variables)], q0))
    setStartSolution(G.EDTracker, vcat(p, λ0))

    while true
		#print(round(α[end], digits=3), ", ")
		q = stepchoice(G, whichstep, α[end], p0, basegradient; initialtime=initialtime, maxseconds=maxseconds)

        _, Tq, vq1, vq2 = get_NTv(q, G, evaluateobjectivefunctiongradient)
		if ( ( Q(q) > Q(p0) - r*α[end]*basegradient'*basegradient ))
			helper = zoom(α[end-1], α[end], Q, evaluateobjectivefunctiongradient,  G, whichstep, p0, basegradient, r, s, initialtime, maxseconds)
			_, Tq, vq1, vq2 = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
			return helper[1], Tq, vq1, vq2, helper[2]
		end
		if ( abs(basegradient'*vq1) <= s*abs(basegradient'*basegradient) )
			return q, Tq, vq1, vq2, α[end]
		end
		if basegradient'*vq1 <= 0
			helper = zoom(α[end], α[end-1], Q, evaluateobjectivefunctiongradient, G, whichstep, p0, basegradient, r, s, initialtime, maxseconds)
			_, Tq, vq1, vq2 = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
			return helper[1], Tq, vq1, vq2, helper[2]
		end
		push!(α, 2*α[end])
		p = q
		deleteat!(α, 1)
		if α[end] >= maxstepsize
			return q, Tq, vq1, vq2, α[end-1]
        end
    end
end

#=
Zoom in on the step lengths between αlo and αhi to find the optimal step size here. This is part of the backtracking line search
=#
function zoom(αlo, αhi, Q, evaluateobjectivefunctiongradient,  G, whichstep, p0, basegradient, r, s, initialtime, maxseconds)
	#qlo = stepchoice(G, whichstep, αlo, p0, basegradient; initialtime=initialtime, maxseconds=maxseconds)
	# To not get stuck in the iteration, we use a for loop instead of a while loop
	# TODO Add a more meaningful stopping criterion
	for _ in 1:8
		global α = 0.5*(αlo+αhi)
		#println("α: ", α)
		global q = stepchoice(G, whichstep, α, p0, basegradient; initialtime=initialtime, maxseconds=maxseconds)
		_, _, vq1, _ = get_NTv(q, G, evaluateobjectivefunctiongradient)

		if  Q(q) > Q(p0) - r*α*basegradient'*basegradient# || Q(q) >= Q(qlo)
			αhi = α
		else
			if Base.abs(basegradient'*vq1) <= Base.abs(basegradient'*basegradient)*s
				return q, α
			end
			if basegradient'*vq1*(αhi-αlo) >= 0
				αhi = αlo
			end
			αlo = α
			#qlo = q
		end
	end
	return q, α
end


#=
 Get the tangent and normal space of a ConstraintVariety at a point q
=#
function get_NTv(q, G::ConstraintVariety, evaluateobjectivefunctiongradient::Function)
    dgq = evaluate.(G.jacobian, G.variables => q)
    Qq = svd(Matrix{Float64}(dgq)).U
	#index = count(p->p>1e-8, S)
    Nq = Qq[:, 1:(G.ambientdimension - G.dimensionofvariety)] # O.N.B. for the normal space at q
    Tq = nullspace(dgq')#(Qq.V)[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # O.N.B. for tangent space at q
    # we evaluate the gradient of the obj fcn at the point `q`
    ∇Qq1, ∇Qq2 = evaluateobjectivefunctiongradient(q)
    w1, w2 = -∇Qq1, -∇Qq2 # direction of decreasing energy function

    vq1 = w1 - Nq * (Nq' * w1) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
	vq2 = w2 - Nq * (Nq' * w2)
	return Nq, Tq, vq1, vq2
end

#=
An object that contains the iteration's information like norms of the projected gradient, step sizes and search directions
=#
struct LocalStepsResult
    initialpoint
    initialstepsize
    allcomputedpoints
    allcomputedprojectedgradientvectors
    allcomputedprojectedgradientvectornorms
    newsuggestedstartpoint
    newsuggestedstepsize
    converged

    function LocalStepsResult(p,ε0,qs,vs,ns,newp,newε0,converged)
        new(p,ε0,qs,vs,ns,newp,newε0,converged)
    end
end

#= Take `maxsteps` steps to try and converge to an optimum. In each step, we use backtracking linesearch
to determine the optimal step size to go along the search direction
WARNING This is redundant and can be merged with findminima
=#
function takelocalsteps(p, ε0, tolerance, G::ConstraintVariety,
                objectiveFunction::Function,
                evaluateobjectivefunctiongradient::Function;
                maxsteps, maxstepsize=2, whichstep="Algorithm 0", initialtime=initialtime, maxseconds=maxseconds)
    _, Tp, vp1, vp2 = get_NTv(p, G, evaluateobjectivefunctiongradient)
    Ts = [Tp] # normal spaces and tangent spaces, columns of Np and Tp are orthonormal bases
    qs, vs, ns = [p], [vp2], [norm(vp1)] # qs=new points on G, vs=projected gradients, ns=norms of projected gradients
    global stepsize = Base.copy(ε0)
    for _ in 1:maxsteps
        q, Tq, vq1, vq2, stepsize = backtracking_linesearch(objectiveFunction, G, evaluateobjectivefunctiongradient, qs[end], Float64(stepsize); whichstep=whichstep, maxstepsize, initialtime=initialtime, maxseconds=maxseconds)
        #print("\n")
		push!(qs, q)
        push!(Ts, Tq)
		length(Ts)>3 ? deleteat!(Ts, 1) : nothing
        push!(ns, norm(vq1))
		#println("ns: ", ns[end])
		push!(vs, vq2)
		length(vs)>3 ? deleteat!(vs, 1) : nothing
        # The next (initial) stepsize is determined by the previous step and how much the energy function changed - in accordance with RieOpt.
		global stepsize = Base.minimum([ Base.maximum([abs(stepsize*vs[end-1]'*evaluateobjectivefunctiongradient(qs[end-1])[2]/(vs[end]'*evaluateobjectivefunctiongradient(qs[end])[2])), 1e-2]), maxstepsize])
    end
    return LocalStepsResult(p,ε0,qs,vs,ns,qs[end],stepsize,(ns[end]<tolerance))
end

#=
 Output object of the method `findminima`
=#
struct OptimizationResult
    computedpoints
    initialpoint
    initialstepsize
    tolerance
    converged
    lastlocalstepsresult
    constraintvariety
    objectivefunction
	lastpointisminimum

    function OptimizationResult(ps,p0,ε0,tolerance,converged,lastLSResult,G,Q,lastpointisminimum)
        new(ps,p0,ε0,tolerance,converged,lastLSResult,G,Q,lastpointisminimum)
    end
end

#=
 The main function of this package. Given an initial point, a tolerance, an objective function and a constraint variety,
 we try to find the objective function's closest local minimum to the initial guess.
=#
function findminima(p0, tolerance,
                G::ConstraintVariety,
                objectiveFunction::Function;
                maxlocalsteps=1, ε0=0.1, whichstep="Algorithm 0", stepdirection = "gradientdescent", maxseconds=1000, initialtime=Base.time())
	#TODO Rework minimality: We are not necessarily at a minimality, if resolveSingularity does not find any better point. => first setequations, then ismin
	#setEquationsAtp!(G,p0)
    p = copy(p0) # initialize before updating `p` below
    ps = [p0] # record the *main steps* from p0, newp, newp, ... until converged
	jacobianG = evaluate.(differentiate(G.fullequations, G.variables), G.variables=>p0)
	evaluateobjectivefunctiongradient = x -> (gradient(objectiveFunction, x), gradient(objectiveFunction, x))
	if stepdirection == "newtonstep"
		evaluateobjectivefunctiongradient = x -> (gradient(objectiveFunction, x), hessian(objectiveFunction, x) \ gradient(objectiveFunction, x))
	end
    _, Tq, v1, v2 = get_NTv(p, G, evaluateobjectivefunctiongradient) # Get the projected gradient at the first point
	# initialize stepsize. Different to RieOpt! Logic: large projected gradient=>far away, large stepsize is admissible.
    lastLSR = LocalStepsResult(p,ε0,[],[],[],p,ε0,false)
    global stepsize = ε0
    while !lastLSR.converged
        if (Base.time()-initialtime)>maxseconds
            return OptimizationResult(ps,p0,ε0,tolerance,false,lastLSR,G,evaluateobjectivefunctiongradient,false)
        end
        # update LSR, only store the *last local run*
        lastLSR = takelocalsteps(p, stepsize, tolerance, G, objectiveFunction, evaluateobjectivefunctiongradient; maxsteps=maxlocalsteps, maxstepsize=1.5, whichstep=whichstep, initialtime=initialtime, maxseconds=maxseconds)
		push!(ps, lastLSR.allcomputedpoints[end])
        p = lastLSR.newsuggestedstartpoint
        global stepsize = lastLSR.newsuggestedstepsize
    end
    _, Tq, v1, v2 = get_NTv(ps[end], G, evaluateobjectivefunctiongradient) # Get the projected gradient at the first point
    if !isMinimum(G,objectiveFunction,evaluateobjectivefunctiongradient,Tq,v1,ps[end])
        return findminima(p0, tolerance, G, objectiveFunction; maxlocalsteps=maxlocalsteps, ε0=ε0, whichstep=whichstep, stepdirection = stepdirection, maxseconds=maxseconds, initialtime=initialtime)
    end

    return OptimizationResult(ps,p0,ε0,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,true)
end

end
