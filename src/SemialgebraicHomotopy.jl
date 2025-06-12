module SemialgebraicHomotopy

import HomotopyContinuation: @var, evaluate, differentiate, start_parameters!, target_parameters!, track!, solve, real_solutions, solutions, solution, rand_subspace, randn, System, ParameterHomotopy, Expression, Tracker, Variable, track, newton
import LinearAlgebra: norm, transpose, qr, rank, normalize, pinv, eigvals, abs, eigvecs, svd, nullspace, zeros
import Plots: plot, scatter!, Animation, frame, cgrad, heatmap, gif, RGBA
import ForwardDiff: hessian, gradient, jacobian
import HomotopyContinuation
import ImplicitPlots: implicit_plot, implicit_plot!

export SemialgebraicSet,
       minimize,
       maximize,
       watch,
       draw,
	   addSamples!,
#INFO: The following package is not maintained by me. Find it here: https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl
	   HomotopyContinuation

#=
 Equips a HomotopyContinuation.Tracker with a start Solution that can be changed on the fly
=#
mutable struct TrackerWithStartSolution
	tracker
	startSolution

	function TrackerWithStartSolution(T::Tracker, startSol::Vector)
		new(T,startSol)
	end
end

function setStartSolution(T::TrackerWithStartSolution, startSol::Vector)
	setfield!(T, :startSolution, startSol)
end


#=
 An object that describes a semialgebraic constraint set by giving its generating equations, inequations, coordinate variables,
 its dimension and its jacobian. Additionally, it contains the system describing the Euclidian Distance Problem
 and samples from the variety.
=#
mutable struct SemialgebraicSet
    variables
    equalities
	inequalities
	fullequations
    fulljacobian
	dimensionofvariety
    samples
	EDTracker

	# Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
	function SemialgebraicSet(variables::Vector{Variable}, equalities::Vector{Expression}, inequalities::Vector{Expression}, d::Int, numsamples::Int)
        fullequations = vcat(equalities,inequalities)
        jacobian = hcat([differentiate(eq, variables) for eq in fullequations]...)
        randL = nothing
		randresult = nothing
        Ωs = []
		if numsamples > 0
			randL = rand_subspace(length(variables); codim=d)
			randResult = solve(equalities; target_subspace = randL, variables=variables, show_progress = true)
		end
        for _ in 1:numsamples
            newΩs = solve(
                    equalities,
                    solutions(randResult);
                    variables = variables,
                    start_subspace = randL,
                    target_subspace = rand_subspace(length(variables); codim = d, real = true),
                    transform_result = (R,p) -> real_solutions(R),
                    flatten = true,
					show_progress = true
            )
            realsols = real_solutions(newΩs)
            push!(Ωs, realsols...)
        end
		Ωs = filter(t -> norm(t)<1e4,Ωs)

		@var u[1:length(variables)]
		@var λ[1:length(equalities)]
		Lagrange = sum((variables-u).^2) + sum(λ.*equalities)
		∇Lagrange = differentiate(Lagrange, vcat(variables,λ))
		EDSystem = System(∇Lagrange, variables=vcat(variables,λ), parameters=u)
		p0 = randn(Float64, length(variables))
		H = ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
		EDTracker = TrackerWithStartSolution(Tracker(H),[])
        new(variables,equalities,inequalities,fullequations,jacobian,d,Ωs,EDTracker)
    end

    # Given implicit equations, sample points from the corresponding variety and return the struct
    function SemialgebraicSet(equalities::Function, inequalities::Function, N::Int, d::Int, numsamples::Int)
        @var variables[1:N]
        algeqnz = equalities(variables)
        algineqnz = inequalities(variables)
		if typeof(algeqnz) != Vector{Expression}
			algeqnz = [algeqnz]
		end
        if typeof(algineqnz) != Vector{Expression}
			algineqnz = [algineqnz]
		end
		SemialgebraicSet(variables, algeqnz, algineqnz,  d, numsamples)
    end

	# Implicit Equations, no sampling, no variables
    function SemialgebraicSet(equalities::Vector{Expression}, inequalities::Vector{Expression}, d::Int)
		F = System(vcat(equalities, inequalities))
        SemialgebraicSet(F.variables, equalities, inequalities, d, 0)
	end

    # HomotopyContinuation-based expressions and variables, no sanples
    function SemialgebraicSet(variables::Vector{Variable}, equalities::Vector{Expression}, inequalities::Vector{Expression}, d::Int)
		SemialgebraicSet(variables, equalities, inequalities, d::Int, 0)
    end

end

#=
Add Samples to an already existing SemialgebraicSet
=#
function addSamples!(G::SemialgebraicSet, newSamples)
	setfield!(G, :samples, vcat(newSamples, G.samples))
end

#=
Compute the system that we need for the onestep and twostep method
=#
function computesystem(p, G::SemialgebraicSet,
                evaluateobjectivefunctiongradient::Function)

    dgp = evaluate.(G.fulljacobian[:,1:length(G.equalities)], G.variables => p)
    Up,_ = qr( transpose(dgp) )
    Np = Up[:, 1:(length(G.variables) - G.dimensionofvariety)] # gives ONB for N_p(G) normal space

    # we evaluate the gradient of the obj fcn at the point `p`
    ∇Qp = evaluateobjectivefunctiongradient(p)[2]

    w = -∇Qp # direction of decreasing energy function
    v = w - Np * (Np' * w) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
	g = G.equalities

    if G.dimensionofvariety > 1 # Need more linear equations when tangent space has dim > 1
        A,_ = qr( hcat(v, Np) )
        A = A[:, (length(G.variables) - G.dimensionofvariety + 1):end] # basis of the orthogonal complement of v inside T_p(G)
        L = A' * G.variables - A' * p # affine linear equations through p, containing v, give curve in variety along v
		u = v / norm(v)
        S = u' * G.variables - u' * (p + Variable(:ε)*u) # create and use the variable ε here.
        F = System( vcat(g,L,S); variables=G.variables, parameters=[Variable(:ε)])
        return F
    else
        u = normalize(v)
        S = u' * G.variables - u' * (p + Variable(:ε)*u) # create and use the variable ε here.
        F = System( vcat(g,S); variables=G.variables, parameters=[Variable(:ε)])
        return F
    end
end

#=
If we are at a point of slow progression / singularity we blow the point up to a sphere and check the intersections (witness sets) with nearby components
for the sample with lowest energy
=#
function resolveSingularity(p, G::SemialgebraicSet, Q::Function, evaluateobjectivefunctiongradient; homotopyMethod=homotopyMethod)
    #q = takelocalsteps(p,5e-3,1e-10,G,Q,evaluateobjectivefunctiongradient; whichstep="gaussnewtonstep").allcomputedpoints[end]
    if Q(q) < Q(p)
        return(q, true) 
    else
        for _ in 1:5
            q = gaussnewtonstep(G, p + 1e-3*randn(Float64, length(p)))[1]
            if Q(q) < Q(p)
                return(q, true) 
            end
        end
    end		
    return(p, false)
end


function _perform_gauss_newton(jac, constraints, variables, p; tol=1e-12, initialtime=Base.time(), maxseconds=100)
    global damping = 0.1
    global q = Base.copy(p)
	global qnew = q
    while length(constraints)>0 && norm(evaluate.(constraints, variables=>q)) > tol
		J = Matrix{Float64}(evaluate.(jac, variables=>q))
        # Randomize the linear system of equations
        stress_dimension = size(nullspace(J; atol=1e-8))[2]
        if stress_dimension > 0
            rand_mat = randn(Float64, length(constraints) - stress_dimension, length(constraints))
            equations = rand_mat*equations
            J = rand_mat*J
        else
            equations = constraints
        end

        # damped Newton's method
        qnew = q - damping * (J' \ evaluate(equations, variables=>q))
        if norm(evaluate(constraints, variables=>qnew)) < norm(evaluate(constraints, variables=>q))
            global damping = damping*1.2
        else
            global damping = damping/2
        end
        if damping < 1e-14 || Base.time()-initialtime > minimum([length(q)/10,maxseconds])
            throw("Newton's method did not converge in time.")
        end
        q = qnew
        if damping > 1
            global damping = 1
        end
	end
	return q, true
end

#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep(G::SemialgebraicSet, p; tol=1e-12, initialtime=Base.time(), maxseconds=100)
    equations = Base.copy(G.equalities)
	jac = G.fulljacobian[:,1:length(G.equalities)]
    q, _ = _perform_gauss_newton(jac, equations, G.variables, p; tol=tol, initialtime=initialtime, maxseconds=maxseconds)

    # A posteriori correction to the inequality constraints
    violated_indices = [i for (i,eq) in enumerate(G.inequalities) if evaluate(eq, G.variables=>q)<=tol]
    new_equations = vcat(G.equalities, G.inequalities[violated_indices])
    jac = G.fulljacobian[:,vcat(1:length(G.equalities), violated_indices)]
    return _perform_gauss_newton(jac, new_equations, G.variables, q; tol=tol, initialtime=initialtime, maxseconds=maxseconds)
end

function EDStep_HC(G::SemialgebraicSet, p, stepsize, v; homotopyMethod, amount_Euler_steps=4)
    #initialtime = Base.time()
    q0 = p#+1e-3*Basenormal[:,1]
    start_parameters!(G.EDTracker.tracker, q0)
    setStartSolution(G.EDTracker, vcat(p, [0. for _ in G.equalities]))
    if homotopyMethod=="HomotopyContinuation"
        q = p+stepsize*v
		target_parameters!(G.EDTracker.tracker, q)
		tracker = track(G.EDTracker.tracker, G.EDTracker.startSolution)
		result = solution(tracker)
        #TODO Implement HC for semialgebraic set as well.
		if all(entry->Base.abs(entry.im)<1e-4, result)
			return gaussnewtonstep(G, [entry.re for entry in result[1:length(p)]])
		else
			return p, false
		end
	else
		global q = p+stepsize*(1/(amount_Euler_steps+1))*v
        global currentSolution = G.EDTracker.startSolution
        global currentSolution, _ = gaussnewtonstep(G, q)
        for step in 1:amount_Euler_steps
            q = p+stepsize*((step+1)/(amount_Euler_steps+1))*v
            global currentSolution, _ = gaussnewtonstep(G, q)
		end
        #println(norm(prev_sol-currentSolution), " ", norm(prediction-currentSolution))
        return currentSolution[1:length(q)], true
	end
end

#=
 Move a line along the projected gradient direction for the length stepsize and calculate the resulting point of intersection with the variety
=#
function onestep(F, p, stepsize)
    solveresult = solve(F, [p]; start_parameters=[0.0], target_parameters=[stepsize],
                                                     show_progress=false)
    sol = real_solutions(solveresult)
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
    solveresult = solve(F, [p]; start_parameters=[0.0 + 0.0im], target_parameters=[midparam], show_progress=false)
    midsols = solutions(solveresult)
    success = false
    if length(midsols) > 0
        midsolution = midsols[1] # only tracked one solution path, thus there should only be one solution
        solveresult = solve(F, [midsolution]; start_parameters=[midparam], target_parameters=[stepsize + 0.0im], show_progress=false)
        realsols = real_solutions(solveresult)
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
function stepchoice(F, constraintset, whichstep, stepsize, p, v; homotopyMethod)
	if(whichstep=="twostep")
		return(twostep(F, p, stepsize))
	elseif whichstep=="onestep"
		return(onestep(F, p, stepsize))
	elseif whichstep=="gaussnewtonstep"||whichstep=="Algorithm 0"
		return(gaussnewtonstep(constraintset, p+stepsize*v))
	elseif whichstep=="gaussnewtonretraction"||whichstep=="newton"||whichstep=="Algorithm 1"
		return(EDStep_HC(constraintset, p, stepsize, v; homotopyMethod="newton"))
	elseif whichstep=="EDStep"||whichstep=="Algorithm 2"
		return(EDStep_HC(constraintset, p, stepsize, v; homotopyMethod))
	else
		throw(error("A step method needs to be provided!"))
	end
end

#=
 Checks, whether p is a local minimum of the objective function Q w.r.t. the tangent space Tp
=#
function isMinimum(G::SemialgebraicSet, Q::Function, evaluateobjectivefunctiongradient, p::Vector; tol=1e-7, criticaltol=1e-3)
	H = hessian(Q, p)
    active_indices = [i for (i,eq) in enumerate(G.fullequations) if isapprox(evaluate(eq, G.variables=>p), 0, atol=tol)]
    active_equations = G.fullequations[active_indices]
    active_jacobian = evaluate(G.fulljacobian[:,active_indices], G.variables=>p)
    if length(active_jacobian)==0 && length(activeindeices)==0
        Tp = nullspace(zeros(Float64,length(G.variables),length(G.variables)))
    else
	    Tp = nullspace(active_jacobian')
    end

    stress_dimension = size(nullspace(active_jacobian; atol=tol))[2]
    # Randomize system to guarantee LICQ
    if stress_dimension > 0
        rand_mat = randn(Float64, length(active_equations) - stress_dimension, length(active_equations))
        active_equations = rand_mat*active_equations
    end

    if length(active_equations)>0
        HConstraints = [evaluate.(differentiate(differentiate(eq, G.variables), G.variables), G.variables=>p) for eq in active_equations]
        # Taylor Approximation of x, since only the Hessian is of interest anyway
        Qalg = Q(p)+(G.variables-p)'*gradient(Q,p)+0.5*(G.variables-p)'*H*(G.variables-p) 
        @var λ[1:length(active_equations)]
        L = Qalg+λ'*active_equations
        ∇L = differentiate(L, vcat(G.variables, λ))
        gL = Matrix{Float64}(evaluate(differentiate(∇L, λ), G.variables=>p))
        bL = -evaluate.(evaluate(∇L,G.variables=>p), λ=>[0 for _ in 1:length(λ)])
        λ0 = map( t-> (t==NaN || t==Inf) ? 0 : t, gL\bL)
        λ0 = any(t->t>0, λ0[length(G.equalities)+1:end]) ? -λ0 : λ0
	    Htotal = H+λ0'*HConstraints
    else
        Htotal = H
    end
	projH = Matrix{Float64}(Tp'*Htotal*Tp)
	projEigvals = real(eigvals(projH)) #projH symmetric => all real eigenvalues
	println("Eigenvalues of the projected Hessian: ", round.(1000 .* projEigvals, sigdigits=3) ./ 1000)
	indices = filter(i->abs(projEigvals[i])<=tol, 1:length(projEigvals))
	projEigvecs = real(eigvecs(projH))[:, indices]
	projEigvecs = Tp*projEigvecs
	if all(q -> q >= tol, projEigvals)
		return true
	elseif any(q -> q <=-tol, projEigvals)
		return false
		#TODO Third derivative at x_0 at proj hessian sing. vectors not 0?!
	# Else take a small step in gradient descent direction and see if the energy decreases
	else
		q = gaussnewtonstep(G, p - 1e-2 * evaluateobjectivefunctiongradient(p)[2]; initialtime=Base.time(), maxseconds=10)[1]
		return Q(q)<Q(p)
	end
end


#=
Use line search with the strong Wolfe condition to find the optimal step length.
This particular method can be found in Nocedal & Wright: Numerical Optimization
=#
function backtracking_linesearch(Q::Function, F::System, G::SemialgebraicSet, evaluateobjectivefunctiongradient::Function, p0::Vector, stepsize::Float64; whichstep="EDStep", maxstepsize=5, initialtime, maxseconds, homotopyMethod="HomotopyContinuation", r=1e-4, s=0.9)
	Basenormal, _, _, basegradient = get_NTv(p0, G, evaluateobjectivefunctiongradient)
	α = [0, stepsize]
	p = Base.copy(p0)
    while true
        try
		    global q, success = stepchoice(F, G, whichstep, α[end], p0, basegradient; homotopyMethod)
        catch e
            @warn e
            maxstepsize = (α[1]+α[2])/2
            α = [α[1], (α[1]+α[2])/2]
            continue
        end
        _, Tq, vq1, vq2 = get_NTv(q, G, evaluateobjectivefunctiongradient)
		if time()-initialtime > maxseconds
			return q, Tq, vq1, vq2, success, α[end]
		end
		if ( ( Q(q) > Q(p0) - r*α[end]*basegradient'*basegradient || (Q(q) > Q(p0) && q!=p0) ) && success)
			helper = zoom(α[end-1], α[end], Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s; initialtime, maxseconds, homotopyMethod)
			_, Tq, vq1, vq2 = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
			return helper[1], Tq, vq1, vq2, helper[2], helper[end]
		end
		if ( abs(basegradient'*vq2) <= s*abs(basegradient'*basegradient) ) && success
			return q, Tq, basegradient, vq2, success, α[end]
		end
		if basegradient'*vq2 <= 0 && success
			helper = zoom(α[end], α[end-1], Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s; initialtime, maxseconds, homotopyMethod)
			_, Tq, vq1, vq2 = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
			return helper[1], Tq, basegradient, vq2, helper[2], helper[end]
		end
		if (success)
			push!(α, 2*α[end])
			p = q
		else
			_, Tp, vp1, vp2 = get_NTv(p, G, evaluateobjectivefunctiongradient)
			return p, Tp, vp1, vp2, success, α[end]
		end
		deleteat!(α, 1)
		if α[end] > maxstepsize
			return q, Tq, vq1, vq2, success, α[end-1]
		end
    end
end

#=
Zoom in on the step lengths between αlo and αhi to find the optimal step size here. This is part of the backtracking line search
=#
function zoom(αlo, αhi, Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s; initialtime, maxseconds, homotopyMethod)
    qlo, suclo = stepchoice(F, G, whichstep, αlo, p0, basegradient; homotopyMethod)
	# To not get stuck in the iteration, we use a for loop instead of a while loop
	# TODO Add a more meaningful stopping criterion
    index = 1
    while index <= 6
		global α = 0.5*(αlo+αhi)
        try
            global q, success = stepchoice(F, G, whichstep, α, p0, basegradient; homotopyMethod)
        catch e
            @warn e
            αlo = αlo < αhi ? αlo : 0.5*(αlo+αhi)
            αhi = αlo < αhi ? 0.5*(αlo+αhi) : αhi
            continue
        end
		_, _, _, vq = get_NTv(q, G, evaluateobjectivefunctiongradient)
		if !success || time()-initialtime > maxseconds
			return q, success, α
		end

		if  Q(q) > Q(p0) - r*α*basegradient'*basegradient
			αhi = α
		else
			if Base.abs(basegradient'*vq) <= Base.abs(basegradient'*basegradient)*s
				return q, success, α
			end
			if basegradient'*vq*(αhi-αlo) >= 0
				αhi = αlo
			end
			αlo = α
			qlo, suclo = q, success
		end
        index += 1
	end
	return q, success, α
end


#=
 Get the tangent and normal space of a SemialgebraicSet at a point q
=#
function get_NTv(q, G::SemialgebraicSet,
                    evaluateobjectivefunctiongradient::Function)
    active_indeices = [i for (i,eq) in enumerate(G.fullequations) if i>length(G.equalities) && abs(evaluate(eq, G.variables=>q)) < 1e-8]
    full_jacobian = evaluate.(G.fulljacobian, G.variables => q)
	active_jacobian = full_jacobian[:,vcat(1:length(G.equalities), active_indeices)]
    ∇Qq1, ∇Qq2 = evaluateobjectivefunctiongradient(q)
    w1, w2 = -∇Qq1, -∇Qq2

    violated_indices = []
    for i in active_indeices
        w1'*full_jacobian[:,i] < 1e-10 ? push!(violated_indices, i) : nothing
    end

    semiactive_jacobian = full_jacobian[:,Vector{Int}(vcat(1:length(G.equalities), violated_indices))]

	try
		global Q_active = svd(Matrix{Float64}(active_jacobian)).U
	catch e
        @warn e
		global Q_active = qr(Matrix{Float64}(active_jacobian)).Q
	end
    try
        global Q_violated = svd(Matrix{Float64}(semiactive_jacobian)).U
    catch e
        @warn e
        global Q_violated = qr(Matrix{Float64}(semiactive_jacobian)).Q
    end

    if length(G.equalities)==0 && length(active_indeices)==0
        return [0. for _ in 1:length(G.variables)], nullspace(zeros(Float64,length(G.variables),length(G.variables))), -∇Qq1, -∇Qq2
    end
	Tq_active = nullspace(active_jacobian')
	Nq_active = Q_active[:, 1:(length(G.variables) - size(Tq_active)[2])] # O.N.B. for the normal space at q
    vq1 = w1 - Nq_active * (Nq_active' * w1) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components

    Tq_violated = nullspace(semiactive_jacobian')
    Nq_violated = Q_violated[:, 1:(length(G.variables) - size(Tq_violated)[2])] # O.N.B. for the normal (half-)space at q
	vq2 = w2 - Nq_violated * (Nq_violated' * w2) # projected gradient -∇Q(p) onto the tangent cone, subtract the normal components
	return Nq_active, Tq_active, vq1, vq2
end

#=
 Parallel transport the vector vj from the tangent space Tj to the tangent space Ti
=#
function paralleltransport(vj, Tj, Ti)
    # transport vj ∈ Tj to become a vector ϕvj ∈ Ti
    # cols(Tj) give ONB for home tangent space, cols(Ti) give ONB for target tangent space
    U,_,Vt = svd( Ti' * Tj )
    Oij = U * Vt' # closest orthogonal matrix to the matrix (Ti' * Tj) comes from svd, remove \Sigma
    ϕvj = Ti * Oij * (Tj' * vj)
    return ϕvj
end

#=
An object that contains the iteration's information like norms of the projected gradient, step sizes and search directions
=#
struct LocalStepsResult
    initialstepsize
    allcomputedpoints
    allcomputedprojectedgradientvectors
    allcomputedprojectedgradientvectornorms
    newsuggestedstartpoint
    newsuggestedstepsize
    converged
    timesturned

    function LocalStepsResult(ε0,qs,vs,ns,newp,newε0,converged,timesturned)
        new(ε0,qs,vs,ns,newp,newε0,converged,timesturned)
    end
end

#= Take `maxsteps` steps to try and converge to an optimum. In each step, we use backtracking linesearch
to determine the optimal step size to go along the search direction
WARNING This is redundant and can be merged with findminima
=#
function takelocalsteps(p::Vector{Float64}, ε0::Float64, tolerance, G::SemialgebraicSet,
                objectiveFunction::Function,
                evaluateobjectivefunctiongradient::Function;
                maxsteps=1, maxstepsize=5, decreasefactor=2.5, initialtime = Base.time(), maxseconds = 100, whichstep="EDStep", homotopyMethod="HomotopyContinuation")
    timesturned, F = 0, System([G.variables[1]])
    _, Tp, vp1, vp2 = get_NTv(p, G, evaluateobjectivefunctiongradient)
    Ts = [Tp] # normal spaces and tangent spaces, columns of Np and Tp are orthonormal bases
    qs, vs, ns = [p], [vp2], [norm(vp2)] # qs=new points on G, vs=projected gradients, ns=norms of projected gradients
    stepsize = Base.copy(ε0)
    for _ in 1:maxsteps
        if Base.time() - initialtime > maxseconds
			break;
        end
		if whichstep=="onestep" || whichstep=="twostep"
        	F = computesystem(qs[end], G, evaluateobjectivefunctiongradient)
		end
        q, Tq, vq1, vq2, success, stepsize = backtracking_linesearch(objectiveFunction, F, G, evaluateobjectivefunctiongradient, qs[end], Float64(stepsize); whichstep, maxstepsize, initialtime, maxseconds, homotopyMethod)
        push!(qs, q); push!(Ts, Tq); push!(ns, norm(vp2)); push!(vs, vq2)
		length(Ts)>3 ? deleteat!(Ts, 1) : nothing
        length(vs)>3 ? deleteat!(vs, 1) : nothing
        if ns[end] < tolerance
            return LocalStepsResult(ε0,qs[2:end],vs,ns,q,stepsize,true,timesturned)
        end
        ϕvj = paralleltransport(vs[end], Ts[end], Ts[end-1])
        if vs[end-1]'*ϕvj < 0
            timesturned += 1
            stepsize = stepsize/decreasefactor
        end
        # The next (initial) stepsize is determined by the previous step and how much the energy function changed - in accordance with RieOpt.
		stepsize = Base.minimum([ Base.maximum([ success ? abs(stepsize*vs[end-1]'*evaluateobjectivefunctiongradient(qs[end-1])[2]/(vs[end]'*evaluateobjectivefunctiongradient(qs[end])[2]))  : 0.1*stepsize, 1e-4]), maxstepsize])
    end
    return LocalStepsResult(ε0,qs[2:end],vs,ns,qs[end],stepsize,false,timesturned)
end

#=
 Output object of the method `findminima`
=#
struct OptimizationResult
    is_minimization
    computedpoints
    initialstepsize
    tolerance
    converged
    lastlocalstepsresult
    constraintvariety
    objectivefunction
	lastpointisoptimum

    function OptimizationResult(is_minimization,ps,ε0,tolerance,converged,lastLSResult,G,Q,lastpointisoptimum)
        new(is_minimization,ps,ε0,tolerance,converged,lastLSResult,G,Q,lastpointisoptimum)
    end
end

#=
 The main function of this package. Given an initial point, a tolerance, an objective function and a constraint variety,
 we try to find the objective function's closest local minimum to the initial guess.
=#
function minimize(p0::Vector{Float64}, tolerance::Float64,
                G::SemialgebraicSet,
                objectiveFunction::Function;
                maxseconds=100, maxlocalsteps=1, initialstepsize=0.1, whichstep="EDStep", initialtime = Base.time(), homotopyMethod = "HomotopyContinuation")
	#TODO Rework minimality: We are not necessarily at a minimality, if resolveSingularity does not find any better point. => first setequations, then ismin
    p = copy(p0) # initialize before updating `p` below
    ps = [p0] # record the *main steps* from p0, newp, newp, ... until converged
	evaluateobjectivefunctiongradient = x -> (gradient(objectiveFunction, x), gradient(objectiveFunction, x))
	# initialize stepsize. Different to RieOpt! Logic: large projected gradient=>far away, large stepsize is admissible.
	global ε0 = initialstepsize
    while (Base.time() - initialtime) <= maxseconds
        # update LSR, only store the *last local run*
        lastLSR = takelocalsteps(p, ε0, tolerance, G, objectiveFunction, evaluateobjectivefunctiongradient; maxsteps=maxlocalsteps, initialtime=initialtime, maxseconds=maxseconds, whichstep=whichstep, homotopyMethod=homotopyMethod)
		global ε0 = lastLSR.newsuggestedstepsize # update and try again!
		append!(ps, lastLSR.allcomputedpoints)
        if lastLSR.converged
			# TODO detect singularities
			if norm(ps[end-1]-ps[end]) < tolerance^3
				optimality = isMinimum(G, objectiveFunction, evaluateobjectivefunctiongradient, ps[end]; criticaltol=tolerance)
				if optimality
					return OptimizationResult(true,ps,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
				end
				println("Resolving")
				p, foundsomething = resolveSingularity(lastLSR.allcomputedpoints[end], G, objectiveFunction, evaluateobjectivefunctiongradient; homotopyMethod=homotopyMethod)
				if foundsomething
					optRes = minimize(p, tolerance, G, objectiveFunction; maxseconds = maxseconds, maxlocalsteps=maxlocalsteps, initialstepsize=initialstepsize, whichstep=whichstep, initialtime=initialtime, homotopyMethod=homotopyMethod)
					return OptimizationResult(true,vcat(ps, optRes.computedpoints),lastLSR.newsuggestedstepsize,tolerance,optRes.lastlocalstepsresult.converged,optRes.lastlocalstepsresult,G,evaluateobjectivefunctiongradient,optRes.lastpointisoptimum)
				end
				return OptimizationResult(true,ps,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
			else
				optimality = isMinimum(G, objectiveFunction, evaluateobjectivefunctiongradient, ps[end]; criticaltol=tolerance)
				if !optimality
					optRes = minimize(ps[end], tolerance, G, objectiveFunction; maxseconds = maxseconds, maxlocalsteps=maxlocalsteps, initialstepsize=initialstepsize, whichstep=whichstep, initialtime=initialtime)
					return OptimizationResult(true,vcat(ps, optRes.computedpoints), lastLSR.newsuggestedstepsize,tolerance,optRes.lastlocalstepsresult.converged,optRes.lastlocalstepsresult,G,evaluateobjectivefunctiongradient,optRes.lastpointisoptimum)
				end
				return OptimizationResult(true,ps,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
			end
        else
            p = lastLSR.newsuggestedstartpoint
            ε0 = lastLSR.newsuggestedstepsize # update and try again!
        end
    end

	display("We ran out of time... Try setting `maxseconds` to a larger value than $(maxseconds)")
	optimality = isMinimum(G, objectiveFunction, evaluateobjectivefunctiongradient, ps[end]; criticaltol=tolerance)
	return OptimizationResult(true,ps,p0,ε0,tolerance,lastLSR.converged,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
end

function maximize(p0, tolerance, G::SemialgebraicSet, objectiveFunction::Function; kwargs...)
    optres = minimize(p0, tolerance, G, x->-objectiveFunction(x); kwargs...)
    return OptimizationResult(false,optres.computedpoints,p0,optres.initialstepsize,tolerance,optres.converged,optres.lastlocalstepresult,G,objectiveFunction,optres.lastpointisoptimum)
end

# Below are functions `watch` and `draw`
# to visualize low-dimensional examples
function watch(result::OptimizationResult; totalseconds=6.0, framesize=nothing, canvas_size=(800,800), sampling_resolution=100,  kwargs...)
    if canvas_size[1] != canvas_size[2]
        @warn "Canvas is expected to be a square."
    end
    ps = result.computedpoints        
	samples = result.constraintvariety.samples
	if !isempty(samples)
		mediannorm = (sort([norm(p) for p in samples]))[Int(floor(samples/2))]
		samples = filter(x -> norm(x) < 2*mediannorm+0.5, samples)
	end
    initplt = plot() # initialize
    M = length(ps)
    framespersecond = M / totalseconds
    if framespersecond > 45
        framespersecond = 45
    end
    startingtime = Base.time()
    dim = length(ps[1])
    anim = Animation()
    if dim == 2
        if framesize==nothing
            fullx = [minimum([q[1] for q in vcat(samples, ps)]) - 0.025, maximum([q[1] for q in vcat(samples, ps)]) + 0.025]
            fully = [minimum([q[2] for q in vcat(samples, ps)]) - 0.025, maximum([q[2] for q in vcat(samples, ps)]) + 0.025]
        else
            if !(framesize isa Union{Vector, Tuple}) || length(framesize)!=2 || any(fr->fr[2]-fr[1]<1e-4, framesize)
                throw(error("Framesize needs to be a tuple of (nonempty) intervals, but is $(typeof(framesize)). Use for example `framesize=((-1.5,1.5),(-1.5,1.5))`."))
            end
            fullx = framesize[1]
            fully = framesize[2]
        end
        initplt = plot([],[],xlims=fullx, ylims=fully, legend=false, size=canvas_size, tickfontsize=16*canvas_size[1]/800, grid=false)

        x_array, y_array = [fullx[1]+i*(fullx[2]-fullx[1])/sampling_resolution for i in 0:sampling_resolution], [fully[1]+j*(fully[2]-fully[1])/sampling_resolution for j in 0:sampling_resolution]
        heatmap_array = [[x_array[i+1], y_array[j+1]] for i in 0:sampling_resolution for j in 0:sampling_resolution]
        for eq in result.constraintvariety.inequalities
            heatmap_array = filter(point->evaluate(eq, result.constraintvariety.variables=>point) >= 0, heatmap_array)
        end
        scatter!(initplt, [ar[1] for ar in heatmap_array], [ar[2] for ar in heatmap_array], markershape=:rect, markersize=4*(100/sampling_resolution)*(canvas_size[1]/800), markerstrokewidth=0, color=RGBA{Float64}(0.75,0.75,0.75))

        for eq in result.constraintvariety.equalities
            implicit_plot!(initplt, x->evaluate(eq, result.constraintvariety.variables=>x))
        end

        if result.lastpointisoptimum
		    initplt = scatter!(initplt, [ps[end][1]], [ps[end][2]], legend=false, markersize=17.5, color=:red, markershape=:rtriangle, xlims=fullx, ylims=fully)
        end
        frame(anim)
        for p in ps[1:end]
            # BELOW: only plot next point, delete older points during animation
            # plt = scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            # BELOW: keep old points during animation.
			initplt = scatter!(initplt, [p[1]], [p[2]], legend=false, markersize=7.5, alpha=0.75, color=:black, xlims=fullx, ylims=fully)
            frame(anim)
        end
        return gif(anim, "optimization_animation_$startingtime.gif", fps=framespersecond)
    else
        throw(error("The only currently supported dimension is 2."))
    end
end

function draw(result::OptimizationResult; kwargs...)
    dim = length(result.computedpoints[1]) # dimension of the ambient space
	ps = result.computedpoints
	samples = result.constraintvariety.samples
	mediannorm = Statistics.median([LinearAlgebra.norm(p) for p in samples])
	# TODO centroid approach rather than mediannorm and then difference from centroid.
	samples = filter(x -> LinearAlgebra.norm(x) < 2*mediannorm+0.5, samples)
    if dim == 2
		fullx = [minimum([q[1] for q in vcat(samples, ps)]) - 0.05, maximum([q[1] for q in vcat(samples, ps)]) + 0.05]
        fully = [minimum([q[2] for q in vcat(samples, ps)]) - 0.05, maximum([q[2] for q in vcat(samples, ps)]) + 0.05]
        g1 = x->evaluate(result.constraintvariety.equalities[1], result.constraintvariety.variables=>x) # should only be a curve in ambient R^2
        plt1 = implicit_plot(g1, xlims=fullx, ylims=fully, legend=false)
        localqs = result.lastlocalstepsresult.allcomputedpoints
        zoomx = [minimum([q[1] for q in localqs]) - 0.05, maximum([q[1] for q in localqs]) + 0.05]
        zoomy = [minimum([q[2] for q in localqs]) - 0.05, maximum([q[2] for q in localqs]) + 0.05]
        plt2 = implicit_plot(g1, xlims=zoomx, ylims=zoomy, legend=false)
        for q in ps
            plt1 = scatter!(plt1, [q[1]], [q[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
        end
        for q in localqs
            plt2 = scatter!(plt2, [q[1]], [q[2]], legend=false, color=:blue, xlims=zoomx, ylims=zoomy)
        end
        vnorms = result.lastlocalstepsresult.allcomputedprojectedgradientvectornorms
        pltvnorms = plot(vnorms, legend=false, title="norm(v) for last local steps")
        plt = plot(plt1,plt2,pltvnorms, layout=(1,3), size=(900,300) )
        return plt
    else
        throw(error("The only currently supported dimension is 2."))
    end
end

end
