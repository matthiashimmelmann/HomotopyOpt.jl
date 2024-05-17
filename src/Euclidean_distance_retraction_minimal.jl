module Euclidean_distance_retraction_minimal

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
       gaussnewtonstep,
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

	function TrackerWithStartSolution(T::Tracker, startSol::Vector, d::Int)
        @var t point[1:d] vector[1:d]
        jacobian_z = hcat([differentiate(eq, T.homotopy.F.interpreted.system.variables) for eq in T.homotopy.F.interpreted.system.expressions]...)
        jacobian_t = [differentiate(eq, t) for eq in evaluate(T.homotopy.F.interpreted.system.expressions, T.homotopy.F.interpreted.system.parameters=>point .+ t .* vector)]
        new(T, startSol, jacobian_z, Vector{Variable}(vector), jacobian_t)
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
    implicitequations
    EDTracker

	# Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
	function ConstraintVariety(varz, eqnz, N::Int, d::Int)
        jacobian = hcat([differentiate(eq, varz) for eq in eqnz]...)
		impliciteq = [p->eqn(varz=>p) for eqn in eqnz]
        randL = nothing
		randresult = nothing

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
		EDTracker = TrackerWithStartSolution(Tracker(H), [], N)
        new(varz,eqnz,fulleqnz,jacobian,N,d,impliciteq,EDTracker)
    end

	# Given implicit equations, sample points from the corresponding variety and return the struct
    function ConstraintVariety(eqnz::Function, N::Int, d::Int)
        @var varz[1:N]
        algeqnz = eqnz(varz)
		if typeof(algeqnz) != Vector{Expression}
			algeqnz = [algeqnz]
		end
		ConstraintVariety(varz, algeqnz, N::Int, d::Int)
    end

	# Implicit Equations, no sampling
    function ConstraintVariety(eqnz,N::Int,d::Int)
		ConstraintVariety(eqnz::Function, N::Int, d::Int)
	end

	#Let the dimension be determined by the algorithm and calculate samples
	function ConstraintVariety(varz,eqnz,p::Vector{Float64})
		G = ConstraintVariety(varz, eqnz, length(varz), 0)
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
function gaussnewtonstep(equations, jacobian, vars, p; tol=1e-12, initialtime=Base.time(), maxtime=10, maxsteps=2)
	global q = p
    global iter = 1
	while norm(evaluate.(equations, vars=>q)) > tol && iter <= maxsteps
        if Base.time()-initialtime > maxtime
            break
        end
		J = Matrix{Float64}(evaluate.(jacobian, vars=>q))
		global q = q .- (J') \ evaluate.(equations, vars=>q)
        global iter += 1
	end
	return q
end

function EDStep(ConstraintVariety, p, v; homotopyMethod, tol=1e-10, amount_Euler_steps=0, maxtime=10)
    initialtime = Base.time()
    Basenormal, _, basegradient = get_NTv(p, ConstraintVariety, v)
    q0 = p+1e-2*Basenormal[:,1]
    start_parameters!(ConstraintVariety.EDTracker.tracker, q0)
    A = evaluate.(differentiate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables[length(p)+1:end]), ConstraintVariety.variables => p)
    λ0 = A\-evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, vcat(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(p, [0 for _ in length(p)+1:length(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables)], q0))
    setStartSolution(ConstraintVariety.EDTracker, vcat(p, λ0))
    #setStartSolution(ConstraintVariety.EDTracker, vcat(p, [0. for _ in λ0]))

    if homotopyMethod=="HomotopyContinuation"
        q = p+v
		target_parameters!(ConstraintVariety.EDTracker.tracker, q)
		tracker = track(ConstraintVariety.EDTracker.tracker, ConstraintVariety.EDTracker.startSolution)
		result = solution(tracker)
		if all(entry->Base.abs(entry.im)<1e-3, result)
			return [entry.re for entry in result[1:length(p)]]
		else
			throw(error("Complex Space entered!"))
		end
	else
        currentSolution = ConstraintVariety.EDTracker.startSolution
        vars = ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.variables
        q = p+(1/(amount_Euler_steps+1))*v
        #currentSolution[1:length(q)] = q
        equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
        currentSolution = currentSolution .+ EulerStep(ConstraintVariety.EDTracker, currentSolution, p, v, 0, 1/(amount_Euler_steps+1))
        currentSolution = gaussnewtonstep(equations, ConstraintVariety.EDTracker.jacobian, vars, currentSolution; initialtime, maxtime, maxsteps = amount_Euler_steps==0 ? 100 : 2)     

        for step in 1:amount_Euler_steps
            q = p+((step+1)/(amount_Euler_steps+1))*v
            #prev_sol = currentSolution
            currentSolution = currentSolution .+ EulerStep(ConstraintVariety.EDTracker, currentSolution, p, v, step/(amount_Euler_steps+1), 1/(amount_Euler_steps+1))
            equations = evaluate(ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.expressions, ConstraintVariety.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
            currentSolution = gaussnewtonstep(equations, ConstraintVariety.EDTracker.jacobian, vars, currentSolution; initialtime, maxtime, maxsteps = amount_Euler_steps==step ? 100 : 2)            
        end
        #println(norm(prev_sol-currentSolution), " ", norm(prediction-currentSolution))
        return currentSolution[1:length(q)]
	end
end

function EulerStep(EDTracker, q, p, v, prev_step, step_size)
    dz = -evaluate.(EDTracker.jacobian, vcat(EDTracker.tracker.homotopy.F.interpreted.system.variables, EDTracker.tracker.homotopy.F.interpreted.system.parameters) => vcat(q, p+prev_step*v))
    du = evaluate.(EDTracker.jacobian_parameter, vcat(EDTracker.tracker.homotopy.F.interpreted.system.variables, EDTracker.ptv) => vcat(q, v))
    return dz \ (du*step_size)
end

#=
 Get the tangent and normal space of a ConstraintVariety at a point q
=#
function get_NTv(q, G::ConstraintVariety, v)
    dgq = evaluate.(G.jacobian, G.variables => q)
    Qq = svd(Matrix{Float64}(dgq)).U
	#index = count(p->p>1e-8, S)
    Nq = Qq[:, 1:(G.ambientdimension - G.dimensionofvariety)] # O.N.B. for the normal space at q
    Tq = nullspace(dgq')#(Qq.V)[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # O.N.B. for tangent space at q
    # we evaluate the gradient of the obj fcn at the point `q`
    ∇Qq1 = v
    w1 = -∇Qq1 # direction of decreasing energy function

    vq1 = w1 - Nq * (Nq' * w1) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
	return Nq, Tq, vq1
end

end