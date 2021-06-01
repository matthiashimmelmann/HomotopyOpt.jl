module HomotopyOpt

import HomotopyContinuation
import LinearAlgebra
import ImplicitPlots: implicit_plot
import Plots
import Statistics
import Implicit3DPlotting: plot_implicit_surface, plot_implicit_surface!, plot_implicit_curve, plot_implicit_curve!#, GLMakiePlottingLibrary
import GLMakie as GLMakiePlottingLibrary
import ForwardDiff

export ConstraintVariety,
       findminima,
       watch,
       draw,
	   addSamples!

# this code modifies `ConstrainedOptimizationByParameterHomotopy.jl`
# so that instead of an ObjectiveFunction, you specify a function called `evaluateobjectivefunctiongradient`
# this makes more sense. You have to define this function yourself,
# but now it does not depend on symbolic algebra from HomotopyContinuation.jl

# TODO add basepoint p+εv to the struct TrackerWithStartSolution, so we only have to solve the ED linear system once
# TODO reduce redundancy

#=
 Equips a HomotopyContinuation.Tracker with a start Solution that can be changed on the fly
=#
mutable struct TrackerWithStartSolution
	tracker
	startSolution
	#basepoint

	function TrackerWithStartSolution(T::HomotopyContinuation.Tracker, startSol::Vector)
		new(T,startSol)
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
    jacobian
    ambientdimension
    dimensionofvariety
    samples
    implicitequations
	EDTracker

	# Given implicit equations, sample points from the corresponding variety and return the struct
    function ConstraintVariety(eqnz::Function, N::Int, d::Int, numsamples::Int)
        HomotopyContinuation.@var varz[1:N]
        algeqnz = [eqn(varz) for eqn in eqnz]
        dg = HomotopyContinuation.differentiate(algeqnz, varz)
        randL = HomotopyContinuation.rand_subspace(N; codim=d)
        randResult = HomotopyContinuation.solve(algeqnz; target_subspace = randL, variables=varz)
        Ωs = []
        for _ in 1:numsamples
            newΩs = HomotopyContinuation.solve(
                    algeqnz,
                    HomotopyContinuation.solutions(randResult);
                    variables = varz,
                    start_subspace = randL,
                    target_subspace = HomotopyContinuation.rand_subspace(N; codim = d, real = true),
                    transform_result = (R,p) -> HomotopyContinuation.real_solutions(R),
                    flatten = true
            )
            realsols = HomotopyContinuation.real_solutions(newΩs)
            push!(Ωs, realsols...)
        end

		HomotopyContinuation.@var u[1:N]
		HomotopyContinuation.@var λ[1:length(eqnz)]
		Lagrange = Base.sum((varz-u).^2) + sum(λ.*eqnz)
		∇Lagrange = HomotopyContinuation.differentiate(Lagrange, vcat(varz,λ))
		EDSystem = HomotopyContinuation.System(∇Lagrange, variables=vcat(varz,λ), parameters=u)
		p0 = HomotopyContinuation.randn(Float64, N)
		H = HomotopyContinuation.ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
		EDTracker = TrackerWithStartSolution(HomotopyContinuation.Tracker(H),[])
        new(varz,algeqnz,dg,N,d,Ωs,eqnz,EDTracker)
    end

	# Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
	function ConstraintVariety(varz, eqnz, N::Int, d::Int, numsamples::Int)
        dg = HomotopyContinuation.differentiate(eqnz, varz)
        randL = HomotopyContinuation.rand_subspace(N; codim=d)
        randResult = HomotopyContinuation.solve(eqnz; target_subspace = randL, variables=varz, show_progress = true)
        Ωs = []
        for _ in 1:numsamples
            newΩs = HomotopyContinuation.solve(
                    eqnz,
                    HomotopyContinuation.solutions(randResult);
                    variables = varz,
                    start_subspace = randL,
                    target_subspace = HomotopyContinuation.rand_subspace(N; codim = d, real = true),
                    transform_result = (R,p) -> HomotopyContinuation.real_solutions(R),
                    flatten = true,
					show_progress = true
            )
            realsols = HomotopyContinuation.real_solutions(newΩs)
            push!(Ωs, realsols...)
        end

		HomotopyContinuation.@var u[1:N]
		HomotopyContinuation.@var λ[1:length(eqnz)]
		Lagrange = Base.sum((varz-u).^2) + sum(λ.*eqnz)
		∇Lagrange = HomotopyContinuation.differentiate(Lagrange, vcat(varz,λ))
		EDSystem = HomotopyContinuation.System(∇Lagrange, variables=vcat(varz,λ), parameters=u)
		p0 = HomotopyContinuation.randn(Float64, N)
		H = HomotopyContinuation.ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
		EDTracker = TrackerWithStartSolution(HomotopyContinuation.Tracker(H),[])
        new(varz,eqnz,dg,N,d,Ωs,eqnz,EDTracker)
    end

	# Implicit Equations, no sampling
    function ConstraintVariety(eqnz,N::Int,d::Int)
        HomotopyContinuation.@var varz[1:N]
        algeqnz = [eqn(varz) for eqn in eqnz]
        dg = HomotopyContinuation.differentiate(algeqnz, varz)

		HomotopyContinuation.@var u[1:N]
		HomotopyContinuation.@var λ[1:length(eqnz)]
		Lagrange = Base.sum((varz-u).^2) + sum(λ.*eqnz)
		∇Lagrange = HomotopyContinuation.differentiate(Lagrange, vcat(varz,λ))
		EDSystem = HomotopyContinuation.System(∇Lagrange, variables=vcat(varz,λ), parameters=u)
		p0 = HomotopyContinuation.randn(Float64, N)
		H = HomotopyContinuation.ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
		EDTracker = TrackerWithStartSolution(HomotopyContinuation.Tracker(H),[])
        new(varz,algeqnz,dg,N,d,[],eqnz,EDTracker)
    end

    # HomotopyContinuation-based expressions and variables, no sanples
    function ConstraintVariety(varz,eqnz,N::Int,d::Int)
        dg = HomotopyContinuation.differentiate(eqnz, varz)
		impliciteq = y->[eqn(varz=>y) for eqn in eqnz]
		HomotopyContinuation.@var u[1:N]
		HomotopyContinuation.@var λ[1:length(eqnz)]
		Lagrange = Base.sum((varz-u).^2) + sum(λ.*eqnz)
		∇Lagrange = HomotopyContinuation.differentiate(Lagrange, vcat(varz,λ))
		EDSystem = HomotopyContinuation.System(∇Lagrange, variables=vcat(varz,λ), parameters=u)
		p0 = HomotopyContinuation.randn(Float64, N)
		H = HomotopyContinuation.ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
		EDTracker = TrackerWithStartSolution(HomotopyContinuation.Tracker(H),[])
        new(varz,eqnz,dg,N,d,[],impliciteq,EDTracker)
    end
end

#=
Add Samples to an already existing ConstraintVariety
=#
function addSamples!(G::ConstraintVariety, newSamples)
	setfield!(G, :samples, vcat(newSamples, G.samples))
end

#=
Compute the system that we need for the onestep and twostep method
=#
function computesystem(p, G::ConstraintVariety,
                evaluateobjectivefunctiongradient::Function)

    dgp = HomotopyContinuation.ModelKit.evaluate(G.jacobian, G.variables => p)
    Up,_ = LinearAlgebra.qr( LinearAlgebra.transpose(dgp) )
    Np = Up[:, 1:(G.ambientdimension - G.dimensionofvariety)] # gives ONB for N_p(G) normal space

    # we evaluate the gradient of the obj fcn at the point `p`
    ∇Qp = evaluateobjectivefunctiongradient(p)
    #display("evaluated the gradient of the objective fcn and got: $∇Qp")

    w = -∇Qp # direction of decreasing energy function
    v = w - Np * (Np' * w) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
	g = G.equations
	#display("variables $(G.variables) and equations $g")

    if G.dimensionofvariety > 1 # Need more linear equations when tangent space has dim > 1
        A,_ = LinearAlgebra.qr( hcat(v, Np))
        A = A[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # basis of the orthogonal complement of v inside T_p(G)
        L = A' * G.variables - A' * p # affine linear equations through p, containing v, give curve in variety along v
		u = v/LinearAlgebra.norm(v)
        S = u' * G.variables - u' * (p + HomotopyContinuation.Variable(:ε)*u) # create and use the variable ε here.
        F = HomotopyContinuation.System( vcat(g,L,S); variables=G.variables,
                                        parameters=[HomotopyContinuation.Variable(:ε)])
        return F
    else
        u = LinearAlgebra.normalize(v)
        S = u' * G.variables - u' * (p + HomotopyContinuation.Variable(:ε)*u) # create and use the variable ε here.
        F = HomotopyContinuation.System( vcat(g,S); variables=G.variables,
                                        parameters=[HomotopyContinuation.Variable(:ε)])
        return F
    end
end

#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep(ConstraintVariety, p, stepsize, v; tol=1e-8)
	q = p+stepsize*v
	jac = Base.hcat([HomotopyContinuation.differentiate(eq,ConstraintVariety.variables) for eq in ConstraintVariety.equations]...)
	while(LinearAlgebra.norm([eq(ConstraintVariety.variables=>q) for eq in ConstraintVariety.equations]) > tol)
		J = jac(ConstraintVariety.variables=>q)
		q = q - LinearAlgebra.transpose(LinearAlgebra.inv(LinearAlgebra.transpose(J)*J)*LinearAlgebra.transpose(J))*[eq(ConstraintVariety.variables=>q) for eq in ConstraintVariety.equations]
	end
	return q, true
end

#=
We predict in the projected gradient direction and correct by solving a Euclidian Distance Problem
=#
function EDStep(ConVar, p, stepsize, v)
	q = p+stepsize*v
	HomotopyContinuation.target_parameters!(ConVar.EDTracker.tracker,q)
	res = HomotopyContinuation.solution(HomotopyContinuation.track(ConVar.EDTracker.tracker, ConVar.EDTracker.startSolution))
	if all(point->Base.abs(point.im)<1e-6, res)
		# TODO set start parameter to be q? Maybe also at a different point?
		return [point.re for point in res[1:length(p)]], true
	else
		return p, false
	end
end

#=
 Move a line along the projected gradient direction for the length stepsize and calculate the resulting point of intersection with the variety
=#
function onestep(F, p, stepsize)
    # we want parameter homotopy from 0.0 to stepsize, so we take two steps
    # first from 0.0 to a complex number parameter, then from that parameter to stepsize.
    solveresult = HomotopyContinuation.solve(F, [p]; start_parameters=[0.0], target_parameters=[stepsize],
                                                     show_progress=false)
    sol = HomotopyContinuation.real_solutions(solveresult)
    success = false
    if length(sol) > 0
        q = sol[1] # only tracked one solution path, thus there should only be one solution
        success = true
    else
        q = p
    end
    return q, success
end

#=
 Similar to onestep. However, we take an intermediate, complex step to avoid singularities
=#
function twostep(F, p, stepsize)
    # we want parameter homotopy from 0.0 to stepsize, so we take two steps
    # first from 0.0 to a complex number parameter, then from that parameter to stepsize.
    midparam = stepsize/2 + stepsize/2*1.0im # complex number *midway* between 0 and stepsize, but off real line
    solveresult = HomotopyContinuation.solve(F, [p]; start_parameters=[0.0 + 0.0im], target_parameters=[midparam],
                                                    show_progress=false)
    midsols = HomotopyContinuation.solutions(solveresult)
    success = false
    if length(midsols) > 0
        midsolution = midsols[1] # only tracked one solution path, thus there should only be one solution
        solveresult = HomotopyContinuation.solve(F, [midsolution]; start_parameters=[midparam],
                                                    target_parameters=[stepsize + 0.0im],
                                                    show_progress=false)
        realsols = HomotopyContinuation.real_solutions(solveresult)
        if length(realsols) > 0
            q = realsols[1] # only tracked one solution path, thus there should only be one solution
            success = true
        else
            q = p
        end
    else
        q = p
    end
    return q, success
end

#=
Determines, which optimization algorithm to use
=#
function stepchoice(F, ConstraintVariety, whichstep, stepsize, p, v)
	if(whichstep=="twostep")
		return(twostep(F, p, stepsize))
	elseif whichstep=="onestep"
		return(onestep(F, p, stepsize))
	elseif whichstep=="gaussnewtonstep"
		return(gaussnewtonstep(ConstraintVariety, p, stepsize, v))
	elseif whichstep=="EDStep"
		return(EDStep(ConstraintVariety, p, stepsize, v))
	else
		throw(error("A step method needs to be provided!"))
	end
end

# WARNING This one is worse than backtracking_linesearch
function alternative_backtracking_linesearch(Q::Function, F::HomotopyContinuation.ModelKit.System, G::ConstraintVariety, evaluateobjectivefunctiongradient::Function, p0::Vector, stepsize::Float64; τ=0.4, r=1e-1, s=0.95, whichstep="twostep")
    α=Base.copy(stepsize)
    p=Base.copy(p0)

	Basenormal, _, basegradient = getNandTandv(p0, G, evaluateobjectivefunctiongradient)
	if whichstep=="EDStep"
		q0 = p+1e-3*Basenormal[:,1]
		HomotopyContinuation.start_parameters!(G.EDTracker.tracker, q0)
		A = HomotopyContinuation.evaluate(HomotopyContinuation.differentiate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end]), G.variables => p)
		λ0 = A\(-HomotopyContinuation.evaluate(HomotopyContinuation.evaluate(HomotopyContinuation.evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end] => [0 for _ in length(p)+1:length(G.EDTracker.tracker.homotopy.F.interpreted.system.variables)]), G.variables => p),  G.EDTracker.tracker.homotopy.F.interpreted.system.parameters=>q0))
		setStartSolution(G.EDTracker, vcat(p,λ0))
	end
    while(true)
		q, success = stepchoice(F, G, whichstep, α, p0, basegradient)
        success ? p=q : nothing
        Nq, Tq, vq = getNandTandv(p, G, evaluateobjectivefunctiongradient)
        # Proceed until the Wolfe condition is satisfied or the stepsize becomes too small. First we quickly find a lower bound, then we gradually increase this lower-bound
		if (Q(p0)-Q(p) >= r*α*Base.abs(basegradient'*evaluateobjectivefunctiongradient(p0)) && vq'*basegradient >= 0 && success)

			for indicator in 1:7
                αsub = α*1.1
				q, success = stepchoice(F, G, whichstep, αsub, p0, basegradient)
                if(!success)
                    return(p, Nq, Tq, vq, false, α)
                end
                Nqsub, Tqsub, vqsub = getNandTandv(q, G, evaluateobjectivefunctiongradient)
                if( Q(p0)-Q(q) < r*αsub*Base.abs(basegradient'*evaluateobjectivefunctiongradient(p0)) || vqsub'*basegradient < 0)
                    return(p, Nq, Tq, vq, true, α)
                elseif( Base.abs(basegradient'*evaluateobjectivefunctiongradient(q)) <= Base.abs(basegradient'*evaluateobjectivefunctiongradient(p0))*s )
                    return(q, Nqsub, Tqsub, vqsub, true, αsub)
                else
                    p=q; α=αsub; vq=vqsub; Tq=Tqsub; Nq=Nqsub;
                end
            end
			return(p, Nq, Tq, vq, true, α)

		elseif α<1e-9
	    	return(p, Nq, Tq, vq, false, stepsize)
        else
            α=τ*α
        end
    end
end

#=
Use line search with the strong Wolfe condition to find the optimal step length.
This particular method can be found in Nocedal & Wright: Numerical Optimization
=#
function backtracking_linesearch(Q::Function, F::HomotopyContinuation.ModelKit.System, G::ConstraintVariety, evaluateobjectivefunctiongradient::Function, p0::Vector, stepsize::Float64, maxstep::Float64; r=1e-2, s=0.8, whichstep="twostep")
	Basenormal, _, basegradient = getNandTandv(p0, G, evaluateobjectivefunctiongradient)
	α0 = 0
	α = [0, stepsize]
	p = Base.copy(p0)
	if whichstep=="EDStep"
		q0 = p+1e-3*Basenormal[:,1]
		HomotopyContinuation.start_parameters!(G.EDTracker.tracker, q0)
		A = HomotopyContinuation.evaluate(HomotopyContinuation.differentiate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end]), G.variables => p)
		λ0 = A\(-HomotopyContinuation.evaluate(HomotopyContinuation.evaluate(HomotopyContinuation.evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end] => [0 for _ in length(p)+1:length(G.EDTracker.tracker.homotopy.F.interpreted.system.variables)]), G.variables => p),  G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q0))
		setStartSolution(G.EDTracker, vcat(p, λ0))
	end

    while true
		q, success = stepchoice(F, G, whichstep, α[end], p0, basegradient)
        Nq, Tq, vq = getNandTandv(q, G, evaluateobjectivefunctiongradient)
		if ( ( Q(q) > Q(p0) + r*α[end]*basegradient'*evaluateobjectivefunctiongradient(p0) || (Q(q) > Q(p) && p!=p0) ) && success)
			helper = zoom(α[end-1], α[end], Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s)
			return helper[1], Nq, Tq, vq, helper[2], helper[end]
		end
		if ( Base.abs(basegradient'*evaluateobjectivefunctiongradient(q)) <= Base.abs(basegradient'*evaluateobjectivefunctiongradient(p0))*s ) && success
			return q, Nq, Tq, vq, success, α[end]
		end
		if basegradient'*evaluateobjectivefunctiongradient(q) >= 0 && success
			helper = zoom(α[end], α[end-1], Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s)
			return helper[1], Nq, Tq, vq, helper[2], helper[end]
		end

		if (success)
			push!(α, 2*(α[end]))
			p = q
		else
			return p, Nq, Tq, vq, success, α[end]
		end
		deleteat!(α, 1)
		if α[end] > maxstep
			return q, Nq, Tq, vq, success, α[end-1]
		end
    end
end

#=
Zoom in on the step lengths between αlo and αhi to find the optimal step size here. This is part of the backtracking line search
=#
function zoom(αlo, αhi, Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s)
	qlo, suclo = stepchoice(F, G, whichstep, αlo, p0, basegradient)
	while true
		α = 0.5*(αlo+αhi)
		q, success = stepchoice(F, G, whichstep, α, p0, basegradient)
		if  Q(q) > Q(p0) + r*α*basegradient'*evaluateobjectivefunctiongradient(p0) || Q(q) >= Q(qlo)
			αhi = α
		else
			if Base.abs(basegradient'*evaluateobjectivefunctiongradient(q)) <= Base.abs(basegradient'*evaluateobjectivefunctiongradient(p0))*s
				return q, success, α
			end
			if basegradient'*evaluateobjectivefunctiongradient(q)*(αhi-αlo) >= 0
				αhi = αlo
			end
			αlo = α
			qlo, suclo = q, success
		end
	end
end


#=
 Get the tangent and normal space of a ConstraintVariety at a point q
=#
function getNandTandv(q, G::ConstraintVariety,
                    evaluateobjectivefunctiongradient::Function)
    dgq = HomotopyContinuation.ModelKit.evaluate(G.jacobian, G.variables => q)
    Qq,_ = LinearAlgebra.qr(transpose(dgq))
    Nq = Qq[:, 1:(G.ambientdimension - G.dimensionofvariety)] # O.N.B. for the normal space at q
    Tq = Qq[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # O.N.B. for tangent space at q

    # we evaluate the gradient of the obj fcn at the point `q`
    ∇Qq = evaluateobjectivefunctiongradient(q)

    w = -∇Qq # direction of decreasing energy function
    vq = w - Nq * (Nq' * w) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components

    return Nq, Tq, vq
end

#=
 Parallel transport the vector vj from the tangent space Tj to the tangent space Ti
=#
function paralleltransport(vj, Tj, Ti)
    # transport vj ∈ Tj to become a vector ϕvj ∈ Ti
    # cols(Tj) give ONB for home tangent space, cols(Ti) give ONB for target tangent space
    U,_,Vt = LinearAlgebra.svd( Ti' * Tj )
    Oij = U * Vt # closest orthogonal matrix to the matrix (Ti' * Tj) comes from svd, remove \Sigma
    ϕvj = Ti * Oij * (Tj' * vj)
    return ϕvj
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
    timesturned
    valleysfound

    function LocalStepsResult(p,ε0,qs,vs,ns,newp,newε0,converged,timesturned,valleysfound)
        new(p,ε0,qs,vs,ns,newp,newε0,converged,timesturned,valleysfound)
    end
end

#= Take `maxsteps` steps to try and converge to an optimum. In each step, we use backtracking linesearch
to determine the optimal step size to go along the search direction
WARNING This is redundant and can be merged with findminima
=#
function takelocalsteps(p, ε0, tolerance, G::ConstraintVariety,
                objectiveFunction::Function,
                evaluateobjectivefunctiongradient::Function;
                maxsteps, decreasefactor=2, initialtime, maxseconds, whichstep="twostep", maxstepsize=100.0)
    timesturned, valleysfound, F = 0, 0, HomotopyContinuation.System([G.variables[1]])
    _, Tp, vp = getNandTandv(p, G, evaluateobjectivefunctiongradient)
    Ts = [Tp] # normal spaces and tangent spaces, columns of Np and Tp are orthonormal bases
    qs, vs, ns = [p], [vp], [LinearAlgebra.norm(vp)] # qs=new points on G, vs=projected gradients, ns=norms of projected gradients
    stepsize = Base.copy(ε0)
    for count in 1:maxsteps
        if Base.time() - initialtime > maxseconds
			break;
        end
		if whichstep=="onestep" || whichstep=="twostep"
        	F = computesystem(qs[end], G, evaluateobjectivefunctiongradient)
		end
        q, Nq, Tq, vq, success, stepsize = backtracking_linesearch(objectiveFunction, F, G, evaluateobjectivefunctiongradient, qs[end], stepsize, maxstepsize; whichstep)
		push!(qs, q)
        push!(Ts, Tq)
		length(Ts)>3 ? deleteat!(Ts, 1) : nothing
        push!(ns, LinearAlgebra.norm(vq))
		push!(vs, vq)
		length(vs)>3 ? deleteat!(vs, 1) : nothing
        if ns[end] < tolerance
            return LocalStepsResult(p,ε0,qs,vs,ns,q,stepsize,true,timesturned,valleysfound)
		# TODO I believe parallel transport is redundant. I think it is already covered by the backtracking linesearch. Not sure though.
        elseif ((ns[end] - ns[end-1]) > 0.0)
            if length(ns) > 2 && ((ns[end-1] - ns[end-2]) < 0.0)
                # projected norms were decreasing, but started increasing!
                # check parallel transport dot product to see if we should slow down
                valleysfound += 1
                ϕvj = paralleltransport(vs[end], Ts[end], Ts[end-2])
                if ((vs[end-2]' * ϕvj) < 0.0)
                    # we think there is a critical point we skipped past! slow down!
                    return LocalStepsResult(p,ε0,qs,vs,ns,qs[end-2],stepsize/decreasefactor,false,timesturned+1,valleysfound)
                end
            end
        end
        # The next (initial) stepsize is determined by the previous step and how much the energy function changed - in accordance with RieOpt.
		stepsize = Base.minimum([ Base.maximum([ success ? stepsize*vs[end-1]'*evaluateobjectivefunctiongradient(qs[end-1])/(vs[end]'*evaluateobjectivefunctiongradient(qs[end]))  : 0.01*stepsize, 0.0001]), maxstepsize])
    end
    return LocalStepsResult(p,ε0,qs,vs,ns,qs[end],stepsize,false,timesturned,valleysfound)
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

    function OptimizationResult(ps,p0,ε0,tolerance,converged,lastLSResult,G,Q)
        new(ps,p0,ε0,tolerance,converged,lastLSResult,G,Q)
    end
end

#=
 The main function of this package. Given an initial point, a tolerance, an objective function and a constraint variety,
 we try to find the objective function's closest local minimum to the initial guess.
=#
function findminima(p0, tolerance,
                G::ConstraintVariety,
                objectiveFunction::Function;
                maxseconds=100, maxlocalsteps=15, initialstepsize=1.0, whichstep="twostep")
    initialtime = Base.time()
    p = copy(p0) # initialize before updating `p` below
    ps = [p0] # record the *main steps* from p0, newp, newp, ... until converged
    evaluateobjectivefunctiongradient = x -> ForwardDiff.gradient(objectiveFunction, x)
    _, _, v = getNandTandv(p0, G, evaluateobjectivefunctiongradient) # Get the projected gradient at the first point
     # initialize stepsize. Different to RieOpt! Logic: large projected gradient=>far away, large stepsize is admissible.
    ε0 = 2*initialstepsize
    lastLSR = LocalStepsResult(p,ε0,[],[],[],p,ε0,false,0,0)
    while (Base.time() - initialtime) <= maxseconds
        # update LSR, only store the *last local run*
        lastLSR = takelocalsteps(p, ε0, tolerance, G, objectiveFunction, evaluateobjectivefunctiongradient; maxsteps=maxlocalsteps, initialtime, maxseconds, whichstep)
        if lastLSR.converged
            push!(ps, lastLSR.newsuggestedstartpoint)
            return OptimizationResult(ps,p0,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient)
        else
            p = lastLSR.newsuggestedstartpoint
            ε0 = lastLSR.newsuggestedstepsize # update and try again!
            push!(ps, p) # record this *main step*
        end
    end

	println("We ran out of time... Try setting `maxseconds` to a larger value than $maxseconds")
    return OptimizationResult(ps,p0,ε0,tolerance,lastLSR.converged,lastLSR,G,evaluateobjectivefunctiongradient)
end

# Below are functions `watch` and `draw`
# to visualize low-dimensional examples
function watch(result::OptimizationResult; totalseconds=5.0)
    ps = result.computedpoints
    samples = result.constraintvariety.samples
    mediannorm = Statistics.median([LinearAlgebra.norm(p) for p in samples])
    samples = filter(x -> LinearAlgebra.norm(x) < 2*mediannorm, samples)
    initplt = Plots.plot() # initialize
    M = length(ps)
    framespersecond = M / totalseconds
    if framespersecond > 45
        framespersecond = 45
    end
    startingtime = Base.time()
    dim = length(ps[1])
    anim = Plots.Animation()
    if dim == 2
        fullx = [minimum([q[1] for q in samples]) - 0.01, maximum([q[1] for q in samples]) + 0.01]
        fully = [minimum([q[2] for q in samples]) - 0.01, maximum([q[2] for q in samples]) + 0.01]
        g1 = result.constraintvariety.equations[1] # should only be a curve in ambient R^2
        initplt = implicit_plot(g1, xlims=fullx, ylims=fully, legend=false)
        Plots.frame(anim)
        for p in ps
            # BELOW: only plot next point, delete older points during animation
            # plt = scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            # BELOW: keep old points during animation.
            initplt = Plots.scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            Plots.frame(anim)
        end
        return Plots.gif(anim, "watch$startingtime.gif", fps=framespersecond)
    elseif dim == 3
        fullx = [minimum([q[1] for q in samples]) - 0.01, maximum([q[1] for q in samples]) + 0.01]
        fully = [minimum([q[2] for q in samples]) - 0.01, maximum([q[2] for q in samples]) + 0.01]
        fullz = [minimum([q[3] for q in samples]) - 0.01, maximum([q[3] for q in samples]) + 0.01]
        g1 = result.constraintvariety.implicitequations[1]
        if(length(result.constraintvariety.implicitequations)>1)
            # should be space curve
            g2 = result.constraintvariety.implicitequations[2]
            initplt = plot_implicit_curve(g1,g2)
        else
            #should be surface
            initplt = plot_implicit_surface(g1)
        end
        pointsys=[GLMakiePlottingLibrary.Point3f0(p) for p in ps]
        GLMakiePlottingLibrary.record(initplt, "watch$startingtime.gif", 1:length(pointsys); framerate = Int64(round(framespersecond))) do i
            GLMakiePlottingLibrary.scatter!(initplt, pointsys[i];
                            color=:black, markersize=20.0)
        end
        return(initplt)
    end
end

function draw(result::OptimizationResult)
    dim = length(result.computedpoints[1]) # dimension of the ambient space
    if dim == 2
        g1 = result.constraintvariety.equations[1] # should only be a curve in ambient R^2
        plt1 = implicit_plot(g1, xlims=(-2,2), ylims=(-2,2), legend=false)
        plt2 = implicit_plot(g1, xlims=(-2,2), ylims=(-2,2), legend=false)
        #f(x,y) = (x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y # replace this with `curve`
        #plt1 = implicit_plot(curve; xlims=(-2,2), ylims=(-2,2), legend=false)
        #plt2 = implicit_plot(curve; xlims=(-2,2), ylims=(-2,2), legend=false)
        globalqs = result.computedpoints
        localqs = result.lastlocalstepsresult.allcomputedpoints
        zoomx = [minimum([q[1] for q in localqs]) - 0.01, maximum([q[1] for q in localqs]) + 0.01]
        zoomy = [minimum([q[2] for q in localqs]) - 0.01, maximum([q[2] for q in localqs]) + 0.01]
        for q in globalqs
            plt1 = Plots.scatter!(plt1, [q[1]], [q[2]], legend=false, color=:black, xlims=[-2,2], ylims=[-2,2])
        end
        for q in localqs
            plt2 = Plots.scatter!(plt2, [q[1]], [q[2]], legend=false, color=:blue, xlims=zoomx, ylims=zoomy)
        end
        vnorms = result.lastlocalstepsresult.allcomputedprojectedgradientvectornorms
        pltvnorms = Plots.plot(vnorms, legend=false, title="norm(v) for last local steps")
        plt = Plots.plot(plt1,plt2,pltvnorms, layout=(1,3), size=(900,300) )
        return plt
    elseif dim == 3
        pointz = result.constraintvariety.samples
        mediannorm = Statistics.median([LinearAlgebra.norm(pt) for pt in pointz])
        pointz = filter(x -> LinearAlgebra.norm(x) < 2*mediannorm, pointz)
        fullx = [minimum([q[1] for q in pointz]) - 0.01, maximum([q[1] for q in pointz]) + 0.01]
        fully = [minimum([q[2] for q in pointz]) - 0.01, maximum([q[2] for q in pointz]) + 0.01]
        fullz = [minimum([q[3] for q in pointz]) - 0.01, maximum([q[3] for q in pointz]) + 0.01]
        globalqs = result.computedpoints
        zoomx = [minimum([q[1] for q in globalqs]) - 0.01, maximum([q[1] for q in globalqs]) + 0.01]
        zoomy = [minimum([q[2] for q in globalqs]) - 0.01, maximum([q[2] for q in globalqs]) + 0.01]
        zoomz = [minimum([q[3] for q in globalqs]) - 0.01, maximum([q[3] for q in globalqs]) + 0.01]

        fig = GLMakiePlottingLibrary.Figure(resolution = (1450, 550))
        ax1 = fig[1, 1] = GLMakiePlottingLibrary.AbstractPlotting.MakieLayout.LScene(fig, width=500, height=500, camera = GLMakiePlottingLibrary.cam3d!, raw = false, limits=GLMakiePlottingLibrary.FRect((fullx[1],fully[1],fullz[1]), (fullx[2]-fullx[1],fully[2]-fully[1],fullz[2]-fullz[1])))
        ax2 = fig[1, 2] = GLMakiePlottingLibrary.AbstractPlotting.MakieLayout.LScene(fig, width=500, height=500, camera = GLMakiePlottingLibrary.cam3d!, raw = false, limits=GLMakiePlottingLibrary.FRect((zoomx[1],zoomy[1],zoomz[1]), (zoomx[2]-zoomx[1],zoomy[2]-zoomy[1],zoomz[2]-zoomz[1])))
        ax3 = fig[1, 3] = GLMakiePlottingLibrary.AbstractPlotting.MakieLayout.Axis(fig, width=300, height=450, title="norm(v) for last local steps")
        g1 = result.constraintvariety.implicitequations[1]
        if(length(result.constraintvariety.implicitequations)>1)
            # should be space curve
            g2 = result.constraintvariety.implicitequations[2]
            plot_implicit_curve!(ax1,g1,g2; xlims=(fullx[1],fullx[2]), ylims=(fully[1],fully[2]), zlims=(fullz[1],fullz[2]))
            plot_implicit_curve!(ax2,g1,g2; xlims=(zoomx[1],zoomx[2]), ylims=(zoomy[1],zoomy[2]), zlims=(zoomz[1],zoomz[2]))
        else
            plot_implicit_surface!(ax1,g1; xlims=(fullx[1],fullx[2]), ylims=(fully[1],fully[2]), zlims=(fullz[1],fullz[2]))
            plot_implicit_surface!(ax2,g1; xlims=(zoomx[1],zoomx[2]), ylims=(zoomy[1],zoomy[2]), zlims=(zoomz[1],zoomz[2]))
        end

        for q in globalqs
            GLMakiePlottingLibrary.scatter!(ax1, GLMakiePlottingLibrary.Point3f0(q);
                legend=false, color=:black, markersize=15)
            GLMakiePlottingLibrary.scatter!(ax2, GLMakiePlottingLibrary.Point3f0(q);
                legend=false, color=:black)
        end

        vnorms = result.lastlocalstepsresult.allcomputedprojectedgradientvectornorms
        GLMakiePlottingLibrary.plot!(ax3,vnorms; legend=false)

        return fig
    end
end

end
