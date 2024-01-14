module HomotopyOpt

import HomotopyContinuation: @var, evaluate, differentiate, start_parameters!, target_parameters!, track!, solve, real_solutions, solutions, solution, rand_subspace, randn, System, ParameterHomotopy, Expression, Tracker, Variable, track
import LinearAlgebra: norm, transpose, qr, rank, normalize, pinv, eigvals, abs, eigvecs, svd
import Plots: plot, scatter!, Animation, frame
import ForwardDiff: hessian, gradient
import ProgressBars: ProgressBar, update

export ConstraintVariety,
       findminima,
       watch,
       draw,
	   addSamples!,
	   setEquationsAtp!

#=
 Equips a HomotopyContinuation.Tracker with a start Solution that can be changed on the fly
=#
mutable struct TrackerWithStartSolution
	tracker
	startSolution
	#basepoint

	function TrackerWithStartSolution(T::Tracker, startSol::Vector)
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
	fullequations
    jacobian
    ambientdimension
	dimensionofvariety
    samples
    implicitequations
	EDTracker

	# Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
	function ConstraintVariety(varz, eqnz, N::Int, d::Int, numsamples::Int)
        dg = differentiate(eqnz, varz)
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
		EDTracker = TrackerWithStartSolution(Tracker(H),[])
        new(varz,eqnz,fulleqnz,dg,N,d,Ωs,impliciteq,EDTracker)
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

	@var u[1:G.ambientdimension]
	@var λ[1:length(eqnz)]
	Lagrange = sum((G.variables-u).^2) + sum(λ.*eqnz)
	∇Lagrange = differentiate(Lagrange, vcat(G.variables,λ))
	EDSystem = System(∇Lagrange, variables=vcat(G.variables,λ), parameters=u)
	H = ParameterHomotopy(EDSystem, start_parameters = p, target_parameters = p)
	EDTracker = TrackerWithStartSolution(Tracker(H),[])
	setfield!(G, :EDTracker, EDTracker)
end

#=
Compute the system that we need for the onestep and twostep method
=#
function computesystem(p, G::ConstraintVariety,
                evaluateobjectivefunctiongradient::Function)

    dgp = evaluate.(G.jacobian, G.variables => p)
    Up,_ = qr( transpose(dgp) )
    Np = Up[:, 1:(G.ambientdimension - G.dimensionofvariety)] # gives ONB for N_p(G) normal space

    # we evaluate the gradient of the obj fcn at the point `p`
    ∇Qp = evaluateobjectivefunctiongradient(p)[2]

    w = -∇Qp # direction of decreasing energy function
    v = w - Np * (Np' * w) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
	g = G.equations

    if G.dimensionofvariety > 1 # Need more linear equations when tangent space has dim > 1
        A,_ = qr( hcat(v, Np) )
        A = A[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # basis of the orthogonal complement of v inside T_p(G)
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
function resolveSingularity(p, G::ConstraintVariety, Q::Function, evaluateobjectivefunctiongradient, whichstep; initialtime = Base.time(), maxseconds = 50)
	if length(p)>5
		q = gaussnewtonstep(G, p, 1e-3, -evaluateobjectivefunctiongradient(p)[2]; initialtime=initialtime, maxseconds=maxseconds)[1]
		( Q(q) < Q(p) && return(q, true) ) || return(q, false)
	end

	eqn = G.fullequations
	var = G.variables
	d = G.dimensionofvariety
	sphereAtPoint = sum((var.-p).^2)-0.0001
	samples = []
	try
		F = System(vcat(eqn,[sphereAtPoint]))
		rel = solve(F; show_progress=false)
		samples = real_solutions(rel)
	catch e
		println("dimension -1: ", e)
	end
	for j in 1:d-1
		try
			a = rand(Float64, length(var), j)
			L = a'*var-a'*p
			F = System( vcat(eqn,[sphereAtPoint],L) )
			append!(samples, real_solutions(solve(F; show_progress=false)))
		catch e
			println("dimension -$(j+1): ", e)
		end
	end

	#TODO Alternative for too large to sample varieties.
	#Random directions? Sampling via Gaussnewton? Gaussnewtonstep altogether?
	minimumvalue = Q(p)
	q = Base.copy(p)
	for sol in samples
		if Q(sol)<minimumvalue
			minimumvalue = Q(sol)
			q = sol
		end
	end

	if q==p && !isempty(samples)
		#In this case, the singularity is optimal in a sense
		return(p,false)
	else
		return(q,true)
	end
end

#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep(G::ConstraintVariety, p, stepsize, v; tol=1e-8, initialtime, maxseconds)
	global q = p+stepsize*v
	global damping = 0.5
	global qnew = q
	jac = hcat([differentiate(eq, G.variables) for eq in G.fullequations]...)
	display(norm(evaluate.(G.fullequations, G.variables=>q)))
	while(norm(evaluate.(G.fullequations, G.variables=>q)) > tol)
		J = Matrix{Float64}(evaluate.(jac, G.variables=>q))
		global qnew = q .- damping*pinv(J)'*evaluate.(G.fullequations, G.variables=>q)
		if norm(evaluate.(G.fullequations, G.variables=>qnew)) <= norm(evaluate.(G.fullequations, G.variables=>q))
			global damping = damping*1.2
		else
			global damping = damping/2
		end
		q = qnew
		if time()-initialtime > maxseconds
			return p, false
		end
	end
	return q, true
end

#=
We predict in the projected gradient direction and correct by solving a Euclidian Distance Problem
=#
function EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod, tol=1e-8)
	q = p+stepsize*v
	if homotopyMethod=="HomotopyContinuation"
		target_parameters!(ConstraintVariety.EDTracker.tracker, q)
		tracker = track(ConstraintVariety.EDTracker.tracker, ConstraintVariety.EDTracker.startSolution)
		result = solution(tracker)
		if all(point->Base.abs(point.im)<1e-4, result)
			return [point.re for point in result[1:length(p)]], true
		else
			return p, false
		end
	elseif homotopyMethod=="Newton"
		currentSolution = vcat(q, ConstraintVariety.EDTracker.startSolution[length(q)+1:end])
		variables = ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables
		equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
		jac = hcat([differentiate(eq, variables) for eq in equations]...)
		while(norm(evaluate.(equations, variables=>currentSolution)) > tol)
			J = evaluate.(jac, variables=>currentSolution)
			currentSolution =  currentSolution .- J \ evaluate.(equations, variables=>currentSolution)
		end
		return currentSolution[1:length(q)], true
	else
		throw(error("Homotopy Method not supported!"))
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
        solveresult = solve(F, [midsolution]; start_parameters=[midparam],
                                                    target_parameters=[stepsize + 0.0im],
                                                    show_progress=false)
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
 Checks, whether p is a local minimum of the objective function Q w.r.t. the tangent space Tp
=#
function isMinimum(G::ConstraintVariety, Q::Function, evaluateobjectivefunctiongradient, Tp, v, p::Vector; tol=1e-4, criticaltol=1e-3)
	if length(p)>20
		q = gaussnewtonstep(G, p, 1e-2, -evaluateobjectivefunctiongradient(p)[2]; initialtime=Base.time(), maxseconds=10)[1]
		return Q(q)<Q(p)
	end

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
	println("Eigenvalues of the projected Hessian: ", round.(1000 .* projEigvals) ./ 1000)
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
		q = gaussnewtonstep(G, p, 1e-2, -evaluateobjectivefunctiongradient(p)[2]; initialtime=Base.time(), maxseconds=10)[1]
		return Q(q)<Q(p)
	end
end

#=
Determines, which optimization algorithm to use
=#
function stepchoice(F, ConstraintVariety, whichstep, stepsize, p, v; initialtime, maxseconds, homotopyMethod)
	if(whichstep=="twostep")
		return(twostep(F, p, stepsize))
	elseif whichstep=="onestep"
		return(onestep(F, p, stepsize))
	elseif whichstep=="gaussnewtonstep"
		return(gaussnewtonstep(ConstraintVariety, p, stepsize, v; initialtime, maxseconds))
	elseif whichstep=="EDStep"
		return(EDStep(ConstraintVariety, p, stepsize, v; homotopyMethod))
	else
		throw(error("A step method needs to be provided!"))
	end
end

# WARNING This one is worse than backtracking_linesearch
function alternative_backtracking_linesearch(Q::Function, F::System, G::ConstraintVariety, evaluateobjectivefunctiongradient::Function, p0::Vector, stepsize::Float64; maxstepsize=100.0, r=1e-4, τ=0.7, whichstep="EDStep", initialtime, maxseconds, homotopyMethod)
    α=Base.copy(stepsize)
    p=Base.copy(p0)

	Basenormal, _, basegradient, _ = get_NTv(p0, G, evaluateobjectivefunctiongradient)
	if whichstep=="EDStep" || homotopyMethod=="Newton"
		q0 = p+1e-3*Basenormal[:,1]
		start_parameters!(G.EDTracker.tracker, q0)
		A = evaluate.(differentiate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end]), G.variables => p)
		λ0 = A\(-evaluate.(evaluate.(evaluate.(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end] => [0 for _ in length(p)+1:length(G.EDTracker.tracker.homotopy.F.interpreted.system.variables)]), G.variables => p),  G.EDTracker.tracker.homotopy.F.interpreted.system.parameters=>q0))
		setStartSolution(G.EDTracker, vcat(p,λ0))
	end
    while(true)
		q, success = stepchoice(F, G, whichstep, α, p0, basegradient; initialtime, maxseconds, homotopyMethod)
        success ? p=q : nothing
        _, Tq, vq1, vq2 = get_NTv(p, G, evaluateobjectivefunctiongradient)
        # Proceed until the Wolfe condition is satisfied or the stepsize becomes too small. First we quickly find a lower bound, then we gradually increase this lower-bound
		if (Q(p0)-Q(p) >= r*α*Base.abs(basegradient'*evaluateobjectivefunctiongradient(p0)[1]) && vq2'*basegradient >= 0 && success)
			return q, Tq, vq1, vq2, success, α
		elseif α<1e-6
	    	return(q, Tq, vq1, vq2, false, stepsize)
        else
            α=τ*α
        end
    end
end

#=
Use line search with the strong Wolfe condition to find the optimal step length.
This particular method can be found in Nocedal & Wright: Numerical Optimization
=#
function backtracking_linesearch(Q::Function, F::System, G::ConstraintVariety, evaluateobjectivefunctiongradient::Function, p0::Vector, stepsize::Float64; whichstep="EDStep", maxstepsize=100.0, initialtime, maxseconds, homotopyMethod="HomotopyContinuation", r=1e-3, s=0.8)
	Basenormal, _, basegradient, _ = get_NTv(p0, G, evaluateobjectivefunctiongradient)
	α = [0, stepsize]
	p = Base.copy(p0)
	if whichstep=="EDStep" || homotopyMethod=="Newton"
		q0 = p+1e-4*Basenormal[:,1]
		start_parameters!(G.EDTracker.tracker, q0)
		A = evaluate.(differentiate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end]), G.variables => p)
		λ0 = A\-evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, vcat(G.EDTracker.tracker.homotopy.F.interpreted.system.variables, G.EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(p, [0 for _ in length(p)+1:length(G.EDTracker.tracker.homotopy.F.interpreted.system.variables)], q0))
		setStartSolution(G.EDTracker, vcat(p, λ0))
	end
	print("α: ")
    while true
		print(round(α[end], digits=3), ", ")
		q, success = stepchoice(F, G, whichstep, α[end], p0, basegradient; initialtime, maxseconds, homotopyMethod)
		if time()-initialtime > maxseconds
			_, Tq, vq1, vq2 = get_NTv(q, G, evaluateobjectivefunctiongradient)
			return q, Tq, vq1, vq2, success, α[end]
		end
        _, Tq, vq1, vq2 = get_NTv(q, G, evaluateobjectivefunctiongradient)
		if ( ( Q(q) > Q(p0) + r*α[end]*basegradient'*basegradient || (Q(q) > Q(p0) && q!=p0) ) && success)
			helper = zoom(α[end-1], α[end], Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s; initialtime, maxseconds, homotopyMethod)
			_, Tq, vq1, vq2 = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
			return helper[1], Tq, vq1, vq2, helper[2], helper[end]
		end
		if ( abs(basegradient'*vq2) <= s*abs(basegradient'*basegradient) ) && success
			return q, Tq, vq1, vq2, success, α[end]
		end
		if basegradient'*vq2 <= 0 && success
			helper = zoom(α[end], α[end-1], Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s; initialtime, maxseconds, homotopyMethod)
			_, Tq, vq1, vq2 = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
			return helper[1], Tq, vq1, vq2, helper[2], helper[end]
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
			return q, Tq, vq1, vq2, success, maxstepsize
		end
    end
end

#=
Zoom in on the step lengths between αlo and αhi to find the optimal step size here. This is part of the backtracking line search
=#
function zoom(αlo, αhi, Q, evaluateobjectivefunctiongradient, F, G, whichstep, p0, basegradient, r, s; initialtime, maxseconds, homotopyMethod)
	qlo, suclo = stepchoice(F, G, whichstep, αlo, p0, basegradient; initialtime, maxseconds, homotopyMethod)
	# To not get stuck in the iteration, we use a for loop instead of a while loop
	# TODO Add a more meaningful stopping criterion
	for _ in 1:8
		global α = 0.5*(αlo+αhi)
		println("α: ", α)
		#println("α: ", α)
		global q, success = stepchoice(F, G, whichstep, α, p0, basegradient; initialtime, maxseconds, homotopyMethod)
		_, _, _, vq2 = get_NTv(q, G, evaluateobjectivefunctiongradient)
		if !success || time()-initialtime > maxseconds
			return q, success, α
		end

		if  Q(q) > Q(p0) + r*α*basegradient'*basegradient || Q(q) >= Q(qlo)
			αhi = α
		else
			if Base.abs(basegradient'*vq2) <= Base.abs(basegradient'*basegradient)*s
				return q, success, α
			end
			if basegradient'*vq2*(αhi-αlo) >= 0
				αhi = αlo
			end
			αlo = α
			qlo, suclo = q, success
		end
	end
	return q, success, α
end


#=
 Get the tangent and normal space of a ConstraintVariety at a point q
=#
function get_NTv(q, G::ConstraintVariety,
                    evaluateobjectivefunctiongradient::Function)
    dgq = evaluate.(G.jacobian, G.variables => q)
    Qq,_ = qr(Matrix{Float64}(transpose(dgq)))
	#index = count(p->p>1e-8, S)
    Nq = Qq[:, 1:(G.ambientdimension - G.dimensionofvariety)] # O.N.B. for the normal space at q
    Tq = Qq[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # O.N.B. for tangent space at q
    # we evaluate the gradient of the obj fcn at the point `q`
    ∇Qq1, ∇Qq2 = evaluateobjectivefunctiongradient(q)
    w1, w2 = -∇Qq1, -∇Qq2 # direction of decreasing energy function

    vq1 = w1 - Nq * (Nq' * w1) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
	vq2 = w2 - Nq * (Nq' * w2)
	return Nq, Tq, vq1, vq2
end

#=
 Parallel transport the vector vj from the tangent space Tj to the tangent space Ti
=#
function paralleltransport(vj, Tj, Ti)
    # transport vj ∈ Tj to become a vector ϕvj ∈ Ti
    # cols(Tj) give ONB for home tangent space, cols(Ti) give ONB for target tangent space
    U,_,Vt = svd( Ti' * Tj )
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
                evaluateobjectivefunctiongradient::Function,
				PBar::ProgressBar;
                maxsteps, maxstepsize=2, decreasefactor=2.2, initialtime, maxseconds, whichstep="EDStep", homotopyMethod="HomotopyContinuation")
    timesturned, valleysfound, F = 0, 0, System([G.variables[1]])
    _, Tp, vp1, vp2 = get_NTv(p, G, evaluateobjectivefunctiongradient)
    Ts = [Tp] # normal spaces and tangent spaces, columns of Np and Tp are orthonormal bases
    qs, vs, ns = [p], [vp2], [norm(vp1)] # qs=new points on G, vs=projected gradients, ns=norms of projected gradients
    stepsize = Base.copy(ε0)
    for _ in 1:maxsteps
        if Base.time() - initialtime > maxseconds
			break;
        end
		if whichstep=="onestep" || whichstep=="twostep"
        	F = computesystem(qs[end], G, evaluateobjectivefunctiongradient)
		end
        q, Tq, vq1, vq2, success, stepsize = backtracking_linesearch(objectiveFunction, F, G, evaluateobjectivefunctiongradient, qs[end], Float64(stepsize); whichstep, maxstepsize, initialtime, maxseconds, homotopyMethod)
		push!(qs, q)
        push!(Ts, Tq)
		length(Ts)>3 ? deleteat!(Ts, 1) : nothing
        push!(ns, norm(vq1))
		println("ns: ", ns[end])
		update(Pbar, ns[end])
		push!(vs, vq2)
		length(vs)>3 ? deleteat!(vs, 1) : nothing
        if ns[end] < tolerance
            return LocalStepsResult(p,ε0,qs,vs,ns,q,stepsize,true,timesturned,valleysfound)
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
		stepsize = Base.minimum([ Base.maximum([ success ? stepsize*vs[end-1]'*evaluateobjectivefunctiongradient(qs[end-1])[2]/(vs[end]'*evaluateobjectivefunctiongradient(qs[end])[2])  : 0.1*stepsize, 1e-4]), maxstepsize])
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
                maxseconds=100, maxlocalsteps=1, initialstepsize=1.0, whichstep="EDStep", initialtime = Base.time(), stepdirection = "gradientdescent", homotopyMethod = "HomotopyContinuation")
	#TODO Rework minimality: We are not necessarily at a minimality, if resolveSingularity does not find any better point. => first setequations, then ismin
	#setEquationsAtp!(G,p0)
	jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p0); atol=tolerance^1.5)
	setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))
    p = copy(p0) # initialize before updating `p` below
    ps = [p0] # record the *main steps* from p0, newp, newp, ... until converged
	jacobianG = evaluate.(differentiate(G.fullequations, G.variables), G.variables=>p0)
	jacRank = rank(jacobianG; atol=tolerance^1.5)
	evaluateobjectivefunctiongradient = x -> (gradient(objectiveFunction, x), gradient(objectiveFunction, x))
	if stepdirection == "newtonstep"
		evaluateobjectivefunctiongradient = x -> (gradient(objectiveFunction, x), hessian(objectiveFunction, x) \ gradient(objectiveFunction, x))
	end
	if jacRank==0
		p, optimality = resolveSingularity(ps[end], G, objectiveFunction, evaluateobjectivefunctiongradient, whichstep; initialtime=initialtime, maxseconds=maxseconds)
		setEquationsAtp!(G, p; tol=tolerance^2)
		jacobianG = evaluate(differentiate(G.fullequations, G.variables), G.variables=>p0)
		jacRank = rank(jacobianG; atol=tolerance^1.5)
	end
    _, Tq, v1, v2 = get_NTv(p, G, evaluateobjectivefunctiongradient) # Get the projected gradient at the first point
	display("Calculated Tangent Space")
	# initialize stepsize. Different to RieOpt! Logic: large projected gradient=>far away, large stepsize is admissible.
	ε0 = 2*initialstepsize
    lastLSR = LocalStepsResult(p,ε0,[],[],[],p,ε0,false,0,0)
	PBar = ProgressBar(0:tol:round(objectiveFunction(p0)/tol)*tol)
    while (Base.time() - initialtime) <= maxseconds
        # update LSR, only store the *last local run*
        lastLSR = takelocalsteps(p, ε0, tolerance, G, objectiveFunction, evaluateobjectivefunctiongradient, PBar; maxsteps=maxlocalsteps, maxstepsize=100., initialtime=initialtime, maxseconds=maxseconds, whichstep=whichstep, homotopyMethod=homotopyMethod)
		push!(ps, lastLSR.allcomputedpoints[end])
		jacobian = evaluate.(differentiate(G.fullequations, G.variables), G.variables=>lastLSR.newsuggestedstartpoint)
		jR = rank(jacobian; atol=tolerance^2)
        if lastLSR.converged
			# if we are in a singularity do a few steps again - if we revert back to the singularity, it is optiomal
			if jR != jacRank || norm(ps[end-1]-ps[end]) < tolerance^2
				#setEquationsAtp!(G, ps[end]; tol=tolerance^1.5)
				jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol=tolerance^2)
				setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))
				_, Tq, v1, _ = get_NTv(ps[end], G, evaluateobjectivefunctiongradient)
				optimality = isMinimum(G, objectiveFunction, evaluateobjectivefunctiongradient, Tq, v1, ps[end]; criticaltol=tolerance)
				if optimality
					return OptimizationResult(ps,p0,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
				end
				println("Resolving")
				p, foundsomething = resolveSingularity(lastLSR.allcomputedpoints[end], G, objectiveFunction, evaluateobjectivefunctiongradient, whichstep; initialtime=initialtime, maxseconds=maxseconds)
				display(norm(p-ps[end]))
				#setEquationsAtp!(G, p; tol=tolerance^1.5)
				jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol=tolerance^2)
				setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))

				if foundsomething
					optRes = findminima(p, tolerance, G, objectiveFunction; maxseconds = maxseconds, maxlocalsteps=maxlocalsteps, initialstepsize=initialstepsize, whichstep=whichstep, initialtime=initialtime, homotopyMethod=homotopyMethod)
					return OptimizationResult(vcat(ps, optRes.computedpoints),p0,lastLSR.newsuggestedstepsize,tolerance,optRes.lastlocalstepsresult.converged,optRes.lastlocalstepsresult,G,evaluateobjectivefunctiongradient,optRes.lastpointisminimum)
				end
				return OptimizationResult(ps,p0,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
			else
				#setEquationsAtp!(G, ps[end]; tol=tolerance^1.5)
				jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol=tolerance^2)
				setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))
				_, Tq, v1, _ = get_NTv(ps[end], G, evaluateobjectivefunctiongradient)
				optimality = isMinimum(G, objectiveFunction, evaluateobjectivefunctiongradient, Tq, v1, ps[end]; criticaltol=tolerance)
				if !optimality
					optRes = findminima(ps[end], tolerance, G, objectiveFunction; maxseconds = maxseconds, maxlocalsteps=maxlocalsteps, initialstepsize=initialstepsize, whichstep=whichstep, initialtime=initialtime)
					return OptimizationResult(vcat(ps, optRes.computedpoints), p0,lastLSR.newsuggestedstepsize,tolerance,optRes.lastlocalstepsresult.converged,optRes.lastlocalstepsresult,G,evaluateobjectivefunctiongradient,optRes.lastpointisminimum)
				end
				return OptimizationResult(ps,p0,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
			end
        else
			# If we are in a point of slow progress or jacobian rank change, we search the neighborhood
			if jR != jacRank || norm(ps[end-1]-ps[end]) < tolerance^2
				#setEquationsAtp!(G, ps[end]; tol=tolerance^1.5)
				jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol=tolerance^2)
				setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))

				_, Tq, v, _ = get_NTv(ps[end], G, evaluateobjectivefunctiongradient)
				optimality = isMinimum(G, objectiveFunction, evaluateobjectivefunctiongradient, Tq, v1, ps[end]; criticaltol=tolerance)
				if optimality
					return OptimizationResult(ps,p0,initialstepsize,tolerance,true,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
				end
				println("Resolving")
				p, foundsomething = resolveSingularity(lastLSR.allcomputedpoints[end], G, objectiveFunction, evaluateobjectivefunctiongradient, whichstep; initialtime=initialtime, maxseconds=maxseconds)
				display(norm(p-ps[end]))
				#setEquationsAtp!(G, p; tol=tolerance^1.5)
				jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol=tolerance^2)
				setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))
				if foundsomething
					maxseconds = maxseconds
					optRes = findminima(p, tolerance, G, objectiveFunction; maxseconds = maxseconds, maxlocalsteps=maxlocalsteps, initialstepsize=initialstepsize, whichstep=whichstep, initialtime=initialtime, homotopyMethod=homotopyMethod)
					return OptimizationResult(vcat(ps, optRes.computedpoints),p0,lastLSR.newsuggestedstepsize,tolerance,optRes.lastlocalstepsresult.converged,optRes.lastlocalstepsresult,G,evaluateobjectivefunctiongradient,optRes.lastpointisminimum)
				end
			else
				p = lastLSR.newsuggestedstartpoint
			end
			jacobian = evaluate.(differentiate(G.equations, G.variables), G.variables=>p)
			jacRank = rank(jacobian; atol=tolerance^1.5)
            ε0 = lastLSR.newsuggestedstepsize # update and try again!
        end
    end

	display("We ran out of time... Try setting `maxseconds` to a larger value than $(maxseconds)")
	p, optimality = resolveSingularity(ps[end], G, objectiveFunction, evaluateobjectivefunctiongradient, whichstep; initialtime=initialtime, maxseconds=maxseconds)
	return OptimizationResult(ps,p0,ε0,tolerance,lastLSR.converged,lastLSR,G,evaluateobjectivefunctiongradient,optimality)
end

# Below are functions `watch` and `draw`
# to visualize low-dimensional examples
function watch(result::OptimizationResult; totalseconds=5.0, fullx = [-1.5,1.5], fully = [-1.5,1.5], fullz = [-1.5,1.5],  kwargs...)
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
		if !isempty(samples)
			fullx = [minimum([q[1] for q in vcat(samples, ps)]) - 0.05, maximum([q[1] for q in vcat(samples, ps)]) + 0.05]
			fully = [minimum([q[2] for q in vcat(samples, ps)]) - 0.05, maximum([q[2] for q in vcat(samples, ps)]) + 0.05]
		end
        g1 = result.constraintvariety.equations[1] # should only be a curve in ambient R^2
        initplt = implicit_plot(g1, xlims=fullx, ylims=fully, legend=false)
		initplt = scatter!(initplt, [ps[end][1]], [ps[end][2]], legend=false, markersize=5, color=:red, xlims=fullx, ylims=fully)
        frame(anim)
        for p in ps
            # BELOW: only plot next point, delete older points during animation
            # plt = scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            # BELOW: keep old points during animation.
			initplt = scatter!(initplt, [p[1]], [p[2]], legend=false, markersize=3.5, color=:black, xlims=fullx, ylims=fully)
            frame(anim)
        end
        return gif(anim, "watch$startingtime.gif", fps=framespersecond)
    elseif dim == 3
		if !isempty(samples)
			fullx = [minimum([q[1] for q in vcat(samples, ps)]) - 0.05, maximum([q[1] for q in vcat(samples, ps)]) + 0.05]
			fully = [minimum([q[2] for q in vcat(samples, ps)]) - 0.05, maximum([q[2] for q in vcat(samples, ps)]) + 0.05]
			fullz = [minimum([q[3] for q in vcat(samples, ps)]) - 0.05, maximum([q[3] for q in vcat(samples, ps)]) + 0.05]
		end
        g1 = result.constraintvariety.implicitequations[1]
		#=
        if(length(result.constraintvariety.implicitequations)>1)
            # should be space curve
            g2 = result.constraintvariety.implicitequations[2]
            initplt = plot_implicit_curve(g1,g2; xlims = (fullx[1], fullx[2]), ylims = (fully[1], fully[2]), zlims = (fullz[1], fullz[2]), kwargs...)
        else
            #should be surface
            initplt = plot_implicit_surface(g1;  xlims = (fullx[1], fullx[2]), ylims = (fully[1], fully[2]), zlims = (fullz[1], fullz[2]), kwargs...)
        end
        pointsys=[GLMakiePlottingLibrary.Point3f0(p) for p in ps]
		GLMakiePlottingLibrary.scatter!(initplt, pointsys[end];
					color=:red, markersize=40.0)
        GLMakiePlottingLibrary.record(initplt, "watch$startingtime.gif", 1:length(pointsys); framerate = Int64(round(framespersecond))) do i
			GLMakiePlottingLibrary.scatter!(initplt, pointsys[i];
	                    color=:black, markersize=30.0)
        end
		=#
        return(initplt)
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
        g1 = result.constraintvariety.equations[1] # should only be a curve in ambient R^2
        plt1 = plot() #implicit_plot(g1, xlims=fullx, ylims=fully, legend=false)
        #f(x,y) = (x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y # replace this with `curve`
        #plt1 = implicit_plot(curve; xlims=(-2,2), ylims=(-2,2), legend=false)
        #plt2 = implicit_plot(curve; xlims=(-2,2), ylims=(-2,2), legend=false)
        localqs = result.lastlocalstepsresult.allcomputedpoints
        zoomx = [minimum([q[1] for q in localqs]) - 0.05, maximum([q[1] for q in localqs]) + 0.05]
        zoomy = [minimum([q[2] for q in localqs]) - 0.05, maximum([q[2] for q in localqs]) + 0.05]
		plt2 = plot()#implicit_plot(g1, xlims=zoomx, ylims=zoomy, legend=false)
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
    elseif dim == 3
		fullx = [minimum([q[1] for q in vcat(samples, ps)]) - 0.05, maximum([q[1] for q in vcat(samples, ps)]) + 0.05]
        fully = [minimum([q[2] for q in vcat(samples, ps)]) - 0.05, maximum([q[2] for q in vcat(samples, ps)]) + 0.05]
        fullz = [minimum([q[3] for q in vcat(samples, ps)]) - 0.05, maximum([q[3] for q in vcat(samples, ps)]) + 0.05]
		localqs = result.lastlocalstepsresult.allcomputedpoints
		zoomx = [minimum([q[1] for q in ps]) - 0.05, maximum([q[1] for q in ps]) + 0.05]
        zoomy = [minimum([q[2] for q in ps]) - 0.05, maximum([q[2] for q in ps]) + 0.05]
        zoomz = [minimum([q[3] for q in ps]) - 0.05, maximum([q[3] for q in ps]) + 0.05]
		#=
        fig = GLMakiePlottingLibrary.Figure(resolution = (1450, 550))
        ax1 = fig[1, 1] = GLMakiePlottingLibrary.AbstractPlotting.MakieLayout.LScene(fig, width=500, height=500, camera = GLMakiePlottingLibrary.cam3d!, raw = false, limits=GLMakiePlottingLibrary.FRect((fullx[1], fully[1], fullz[1]), (fullx[2]-fullx[1], fully[2]-fully[1], fullz[2]-fullz[1])))
        ax2 = fig[1, 2] = GLMakiePlottingLibrary.AbstractPlotting.MakieLayout.LScene(fig, width=500, height=500, camera = GLMakiePlottingLibrary.cam3d!, raw = false, limits=GLMakiePlottingLibrary.FRect((zoomx[1], zoomy[1], zoomz[1]), (zoomx[2]-zoomx[1], zoomy[2]-zoomy[1], zoomz[2]-zoomz[1])))
        ax3 = fig[1, 3] = GLMakiePlottingLibrary.AbstractPlotting.MakieLayout.Axis(fig, width=300, height=450, title="norm(v) for last local steps")
        g1 = result.constraintvariety.implicitequations[1]
        if(length(result.constraintvariety.implicitequations)>1)
            # should be space curve
            g2 = result.constraintvariety.implicitequations[2]
            plot_implicit_curve!(ax1,g1,g2; xlims=(fullx[1],fullx[2]), ylims=(fully[1],fully[2]), zlims=(fullz[1],fullz[2]), kwargs...)
            plot_implicit_curve!(ax2,g1,g2; xlims=(zoomx[1],zoomx[2]), ylims=(zoomy[1],zoomy[2]), zlims=(zoomz[1],zoomz[2]), kwargs...)
        else
            plot_implicit_surface!(ax1,g1; xlims=(fullx[1],fullx[2]), ylims=(fully[1],fully[2]), zlims=(fullz[1],fullz[2]), kwargs...)
            plot_implicit_surface!(ax2,g1; xlims=(zoomx[1],zoomx[2]), ylims=(zoomy[1],zoomy[2]), zlims=(zoomz[1],zoomz[2]), kwargs...)
        end

        for q in ps
            GLMakiePlottingLibrary.scatter!(ax1, GLMakiePlottingLibrary.Point3f0(q);
                legend=false, color=:black, markersize=15)
            GLMakiePlottingLibrary.scatter!(ax2, GLMakiePlottingLibrary.Point3f0(q);
                legend=false, color=:black)
        end

        vnorms = result.lastlocalstepsresult.allcomputedprojectedgradientvectornorms
        GLMakiePlottingLibrary.plot!(ax3,vnorms; legend=false)
        return fig
		=#
    end
end

end
