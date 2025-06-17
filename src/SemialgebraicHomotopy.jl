module SemialgebraicHomotopy

import HomotopyContinuation:
    @var,
    evaluate,
    differentiate,
    start_parameters!,
    target_parameters!,
    track!,
    solve,
    real_solutions,
    solutions,
    solution,
    rand_subspace,
    randn,
    System,
    ParameterHomotopy,
    Expression,
    Tracker,
    Variable,
    track,
    newton
import LinearAlgebra:
    norm, transpose, qr, rank, normalize, pinv, eigvals, abs, eigvecs, svd, nullspace, zeros
import Plots: plot, scatter!, Animation, frame, cgrad, heatmap, gif, RGBA
using Plots.PlotMeasures
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
    tracker::Any
    startSolution::Any

    function TrackerWithStartSolution(T::Tracker, startSol::Vector)
        new(T, startSol)
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
    variables::Any
    equalities::Any
    inequalities::Any
    fullequations::Any
    fulljacobian::Any
    dimensionofvariety::Any
    samples::Any
    EDTracker::Any
    full_EDSystem::Any

    function Base.show(io::IO, G::SemialgebraicSet)
        print("$(typeof(G))(variables: $(G.variables), equalities: $(G.equalities), inequalities: $(G.inequalities), dimensionofvariety: $(G.dimensionofvariety)$(isempty(G.samples) ? ")" : ", samples: $(G.samples))")")
    end

    # Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
    function SemialgebraicSet(
        variables::Vector{Variable},
        equalities::Vector{Expression},
        inequalities::Vector{Expression},
        d::Int,
        numsamples::Int,
    )
        fullequations = vcat(equalities, inequalities)
        jacobian = hcat([differentiate(eq, variables) for eq in fullequations]...)
        randL = nothing
        randresult = nothing
        Ωs = []
        if numsamples > 0
            randL = rand_subspace(length(variables); codim = d)
            randResult = solve(
                equalities;
                target_subspace = randL,
                variables = variables,
                show_progress = true,
            )
        end
        for _ = 1:numsamples
            newΩs = solve(
                equalities,
                solutions(randResult);
                variables = variables,
                start_subspace = randL,
                target_subspace = rand_subspace(length(variables); codim = d, real = true),
                transform_result = (R, p) -> real_solutions(R),
                flatten = true,
                show_progress = true,
            )
            realsols = real_solutions(newΩs)
            push!(Ωs, realsols...)
        end
        Ωs = filter(t -> norm(t)<1e4, Ωs)

        #TODO Only compute EDSystem once
        @var u[1:length(variables)]
        @var λ[1:length(equalities)]
        Lagrange = sum((variables-u) .^ 2) + sum(λ .* equalities)
        ∇Lagrange = differentiate(Lagrange, vcat(variables, λ))
        EDSystem = System(∇Lagrange, variables = vcat(variables, λ), parameters = u)
        p0 = randn(Float64, length(variables))
        H = ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
        EDTracker = TrackerWithStartSolution(Tracker(H), [])

        @var μ[1:(length(equalities)+length(inequalities))]
        full_Lagrange = sum((variables-u) .^ 2) + sum(λ .* equalities)
        full_∇Lagrange = differentiate(Lagrange, vcat(variables, μ))
        full_EDSystem = System(∇Lagrange, variables = vcat(variables, λ), parameters = u)
        new(
            variables,
            equalities,
            inequalities,
            fullequations,
            jacobian,
            d,
            Ωs,
            EDTracker,
            full_EDSystem,
        )
    end

    # Given implicit equations, sample points from the corresponding variety and return the struct
    function SemialgebraicSet(
        equalities::Function,
        inequalities::Function,
        N::Int,
        d::Int,
        numsamples::Int,
    )
        @var variables[1:N]
        algeqnz = equalities(variables)
        algineqnz = inequalities(variables)
        if typeof(algeqnz) != Vector{Expression}
            algeqnz = [algeqnz]
        end
        if typeof(algineqnz) != Vector{Expression}
            algineqnz = [algineqnz]
        end
        SemialgebraicSet(variables, algeqnz, algineqnz, d, numsamples)
    end

    # Implicit Equations, no sampling, no variables
    function SemialgebraicSet(
        equalities::Vector{Expression},
        inequalities::Vector{Expression},
        d::Int,
    )
        F = System(vcat(equalities, inequalities))
        SemialgebraicSet(F.variables, equalities, inequalities, d, 0)
    end

    # HomotopyContinuation-based expressions and variables, no sanples
    function SemialgebraicSet(
        variables::Vector{Variable},
        equalities::Vector{Expression},
        inequalities::Vector{Expression},
        d::Int,
    )
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
If we are at a point of slow progression / singularity we blow the point up to a sphere and check the intersections (witness sets) with nearby components
for the sample with lowest energy
=#
function resolveSingularity(
    p,
    G::SemialgebraicSet,
    Q::Function,
    evaluateobjectivefunctiongradient;
    homotopyMethod = homotopyMethod,
    traditional_newton = true,
)
    if Q(q) < Q(p)
        return (q, true)
    else
        for _ = 1:5
            q = gaussnewtonstep(
                G,
                p + 1e-3*randn(Float64, length(p));
                traditional_newton = traditional_newton,
            )[1]
            if Q(q) < Q(p)
                return (q, true)
            end
        end
    end
    return (p, false)
end

#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep_lagrange(
    G::SemialgebraicSet,
    p,
    reference_point;
    tol = 1e-13,
    initialtime = Base.time(),
    maxseconds = 100,
    maxsteps = 50,
)
    F = G.full_EDSystem
    cur_p = vcat(p, [0 for _ = (length(p)+1):length(F.variables)])
    # A posteriori correction to the inequality constraints
    violated_indices =
        [i for (i, eq) in enumerate(G.inequalities) if evaluate(eq, G.variables=>q)<=-tol]
    new_equations = vcat(G.equalities, G.inequalities[violated_indices])
    new_F = System(new_equations, variables = F.variables, parameters = F.parameters)
    while !isempty(violated_indices)
        newton_result = newton(new_F, cur_p, reference_point; max_iters = maxsteps)
        if newton_result.return_code != :success
            return cur_p, false
        end
        cur_p = real.(newton_result.x)
        violated_indices = [
            i for (i, eq) in enumerate(G.inequalities) if
            evaluate(eq, G.variables=>cur_p)<=-tol
        ]
        new_equations = vcat(G.equalities, G.inequalities[violated_indices])
        new_F = System(new_equations, variables = F.variables, parameters = F.parameters)
    end
    return cur_p, true
end


function traditional_newton_correct(
    equations,
    jacobian,
    variables,
    p;
    tol = 1e-13,
    maxsteps = 50,
)
    global q = Base.copy(p)
    global qnew = q
    new_equations = Base.copy(equations)
    while length(equations)>0 && norm(evaluate.(equations, variables=>q), Inf) > tol
        # damped Newton's method
        J = Matrix{Float64}(evaluate.(jacobian, G.variables=>q))
        # Randomize the linear system of equations
        stress_dimension = size(nullspace(J; atol = 1e-8))[2]
        if stress_dimension > 0
            rand_mat =
                randn(Float64, length(equations) - stress_dimension, length(equations))
            new_equations = rand_mat*equations
            J = rand_mat*J
        else
            new_equations = equations
        end
        qnew = q - damping * (J' \ evaluate(new_equations, G.variables=>q))
        if norm(evaluate(equations, variables=>qnew), Inf) <
           norm(evaluate(equations, G.variables=>q), Inf)
            global damping = damping*1.2
        else
            global damping = damping/2
        end
        if damping < 1e-14 || Base.time()-initialtime > minimum([length(q)/3, maxseconds])
            throw("Newton's method did not converge in time.")
        end
        q = qnew
        if damping > 1
            global damping = 1
        end
    end
    return q
end


function symmetric_newton_correct(
    equations,
    jacobian,
    variables,
    p;
    tol = 1e-13,
    maxsteps = 50,
)
    global q = Base.copy(p)
    global qnew = q
    new_equations = Base.copy(equations)
    J = Matrix{Float64}(evaluate.(jacobian, G.variables=>q))
    # Randomize the linear system of equations
    stress_dimension = size(nullspace(J; atol = 1e-8))[2]
    if stress_dimension > 0
        rand_mat = randn(Float64, length(equations) - stress_dimension, length(equations))
        new_equations = rand_mat*equations
        J = rand_mat*J
    else
        new_equations = equations
    end
    while length(equations)>0 && norm(evaluate.(equations, variables=>q), Inf) > tol
        # damped Newton's method
        qnew = q - damping * (J' \ evaluate(new_equations, G.variables=>q))
        if norm(evaluate(equations, variables=>qnew), Inf) <
           norm(evaluate(equations, G.variables=>q), Inf)
            global damping = damping*1.2
        else
            global damping = damping/2
        end
        if damping < 1e-14 || Base.time()-initialtime > minimum([length(q)/3, maxseconds])
            throw("Newton's method did not converge in time.")
        end
        q = qnew
        if damping > 1
            global damping = 1
        end
    end
    return q
end

#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep(
    G::SemialgebraicSet,
    p;
    tol = 1e-13,
    initialtime = Base.time(),
    maxseconds = 100,
    traditional_newton = true,
)
    jac = G.fulljacobian[:, 1:length(G.equalities)]
    global damping = 0.2
    global q =
        traditional_newton ?
        traditional_newton_correct(G.equalities, jac, G.variables, p; tol = tol) :
        symmetric_newton_correct(G.equalities, jac, G.variables, p; tol = tol)

    # A posteriori correction to the inequality constraints
    global damping = 0.2
    violated_indices =
        [i for (i, eq) in enumerate(G.inequalities) if evaluate(eq, G.variables=>q)<=-tol]
    new_equations = vcat(G.equalities, G.inequalities[violated_indices])
    jac = G.fulljacobian[:, vcat(1:length(G.equalities), violated_indices)]
    while length(violated_indices) > 0
        global q =
            traditional_newton ?
            traditional_newton_correct(new_equations, jac, G.variables, q; tol = tol) :
            symmetric_newton_correct(new_equations, jac, G.variables; tol = tol)
        violated_indices = [
            i for (i, eq) in enumerate(G.inequalities) if evaluate(eq, G.variables=>q)<=-tol
        ]
        new_equations = vcat(G.equalities, G.inequalities[violated_indices])
        jac = G.fulljacobian[:, vcat(1:length(G.equalities), violated_indices)]
    end
    return q, true
end


function EDStep_HC(
    G::SemialgebraicSet,
    p,
    stepsize,
    v;
    homotopyMethod,
    amount_Euler_steps = 3,
    traditional_newton = true,
)
    #initialtime = Base.time()
    q0 = p#+1e-3*Basenormal[:,1]
    start_parameters!(G.EDTracker.tracker, q0)
    setStartSolution(G.EDTracker, vcat(p, [0.0 for _ in G.equalities]))
    if homotopyMethod=="HomotopyContinuation"||homotopyMethod=="Algorithm 2"
        q = p+stepsize*v
        target_parameters!(G.EDTracker.tracker, q)
        tracker = track(G.EDTracker.tracker, G.EDTracker.startSolution)
        result = solution(tracker)
        #TODO Implement HC for semialgebraic set as well.
        if all(entry->Base.abs(entry.im)<1e-6, result)
            try
                q, success = gaussnewtonstep_lagrange(G, [entry.re for entry in result], q)
                if !success
                    throw(
                        error(
                            "Newton's method did not converge! Trying smaller incremental steps.",
                        ),
                    )
                end
                return q, true
            catch e
                @warn e
                q0 = Base.copy(p)
                for i = 1:amount_Euler_steps
                    setStartSolution(G.EDTracker, vcat(q0, [0.0 for _ in G.equalities]))
                    start_parameters!(G.EDTracker.tracker, q0)
                    q = q0+(1/amount_Euler_steps)*stepsize*v
                    target_parameters!(G.EDTracker.tracker, q)
                    tracker = track(G.EDTracker.tracker, G.EDTracker.startSolution)
                    cur_result = solution(tracker)
                    if all(entry->Base.abs(entry.im)<1e-6, cur_result)
                        q, success = gaussnewtonstep_lagrange(
                            G,
                            [entry.re for entry in cur_result],
                            q,
                        )
                        if !success
                            return q0, false
                        end
                        q0 = q
                    else
                        return q0, false
                    end
                end
            end
        else
            return p, false
        end
    else
        global q = p+stepsize*(1/(amount_Euler_steps+1))*v
        global currentSolution = G.EDTracker.startSolution
        global currentSolution, _ =
            gaussnewtonstep(G, q; traditional_newton = traditional_newton)
        for step = 1:amount_Euler_steps
            q = p+stepsize*((step+1)/(amount_Euler_steps+1))*v
            global currentSolution, _ =
                gaussnewtonstep(G, q; traditional_newton = traditional_newton)
        end
        return currentSolution[1:length(q)], true
    end
end

#=
Determines, which optimization algorithm to use
=#
function stepchoice(constraintset, whichstep, stepsize, p, v; traditional_newton = true)
    if whichstep=="gaussnewtonstep" || whichstep=="Algorithm 0"
        return (gaussnewtonstep(
            constraintset,
            p+stepsize*v;
            traditional_newton = traditional_newton,
        ))
    elseif whichstep=="gaussnewtonretraction"||whichstep=="newton"||whichstep=="Algorithm 1"
        return (EDStep_HC(constraintset, p, stepsize, v; homotopyMethod = "newton"))
    elseif whichstep=="EDStep"||whichstep=="Algorithm 2"
        return (EDStep_HC(
            constraintset,
            p,
            stepsize,
            v;
            homotopyMethod = "HomotopyContinuation",
        ))
    else
        throw(error("A step method needs to be provided!"))
    end
end

#=
 Checks, whether p is a local minimum of the objective function Q w.r.t. the tangent space Tp
=#
function isMinimum(
    G::SemialgebraicSet,
    Q::Function,
    evaluateobjectivefunctiongradient,
    p::Vector;
    tol = 1e-4,
    criticaltol = 1e-3,
    traditional_newton = true,
)
    H = hessian(Q, p)
    active_indices = [
        i for (i, eq) in enumerate(G.fullequations) if
        isapprox(evaluate(eq, G.variables=>p), 0, atol = tol)
    ]
    active_equations = G.fullequations[active_indices]
    if length(active_indices)==0
        Tp = nullspace(zeros(Float64, length(G.variables), length(G.variables)))
    else
        active_jacobian = evaluate(G.fulljacobian[:, active_indices], G.variables=>p)
        Tp = nullspace(active_jacobian')
        stress_dimension = size(nullspace(active_jacobian; atol = tol))[2]
        # Randomize system to guarantee LICQ
        if stress_dimension > 0
            rand_mat = randn(
                Float64,
                length(active_equations) - stress_dimension,
                length(active_equations),
            )
            active_equations = rand_mat*active_equations
        end
    end


    if length(active_equations)>0
        HConstraints = [
            evaluate.(
                differentiate(differentiate(eq, G.variables), G.variables),
                G.variables=>p,
            ) for eq in active_equations
        ]
        # Taylor Approximation of x, since only the Hessian is of interest anyway
        Qalg = Q(p)+(G.variables-p)'*gradient(Q, p)+0.5*(G.variables-p)'*H*(G.variables-p)
        @var λ[1:length(active_equations)]
        L = Qalg+λ'*active_equations
        ∇L = differentiate(L, vcat(G.variables, λ))
        gL = Matrix{Float64}(evaluate(differentiate(∇L, λ), G.variables=>p))
        bL = -evaluate.(evaluate(∇L, G.variables=>p), λ=>[0 for _ = 1:length(λ)])
        λ0 = map(t->(t==NaN || t==Inf) ? 0 : t, gL\bL)
        λ0 = any(t->t>0, λ0[(length(G.equalities)+1):end]) ? -λ0 : λ0
        Htotal = H+λ0'*HConstraints
    else
        Htotal = H
    end
    projH = Matrix{Float64}(Tp'*Htotal*Tp)
    projEigvals = real(eigvals(projH)) #projH symmetric => all real eigenvalues
    indices = filter(i->abs(projEigvals[i])<=tol, 1:length(projEigvals))
    projEigvecs = real(eigvecs(projH))[:, indices]
    projEigvecs = Tp*projEigvecs
    if all(q -> q >= tol, projEigvals)
        return true
    elseif any(q -> q <= -tol, projEigvals)
        return false
        #TODO Third derivative at x_0 at proj hessian sing. vectors not 0?!
        # Else take a small step in gradient descent direction and see if the energy decreases
    else
        q = gaussnewtonstep(
            G,
            p - 1e-2 * evaluateobjectivefunctiongradient(p);
            initialtime = Base.time(),
            maxseconds = 10,
            traditional_newton = true,
        )[1]
        return Q(q)<Q(p)
    end
end

#=
The momentum strategy only relies on an Armijo backtracking linesearch
=#
function backtracking_linesearch_momentum(
    Q::Function,
    G::SemialgebraicSet,
    evaluateobjectivefunctiongradient::Function,
    p0::Vector,
    stepsize::Float64;
    whichstep = "EDStep",
    maxstepsize = 5,
    initialtime,
    maxseconds,
    homotopyMethod = "HomotopyContinuation",
    r = 1e-4,
    s = 0.9,
    traditional_newton = true,
)
    _, _, basegradient = get_NTv(p0, G, evaluateobjectivefunctiongradient)
    α = [stepsize]
    p = Base.copy(p0)
    while length(α) <= 7 && time()-initialtime <= maxseconds
        try
            global q, success = stepchoice(
                G,
                whichstep,
                α[end],
                p0,
                basegradient;
                traditional_newton =  traditional_newton=traditional_newton ,
            )
        catch e
            @warn e
            maxstepsize = α[end]/2
            push!(α, α[end]/2)
            continue
        end
        if (Q(q) <= Q(p0) - r*α[end]*basegradient'*basegradient && success)
            _, Tq, vq = get_NTv(q, G, evaluateobjectivefunctiongradient)
            return q, Tq, vq, success, 1.25*α[end]
        end
        if (success)
            push!(α, α[end]/2)
            p = q
        else
            _, Tp, vp = get_NTv(p, G, evaluateobjectivefunctiongradient)
            return p, Tp, vp, success, α[end]
        end
    end
    _, Tq, vq = get_NTv(q, G, evaluateobjectivefunctiongradient)
    return q, Tq, vq, success, α[end]
end


#=
Use line search with the strong Wolfe condition to find the optimal step length.
This particular method can be found in Nocedal & Wright: Numerical Optimization
=#
function backtracking_linesearch(
    Q::Function,
    G::SemialgebraicSet,
    evaluateobjectivefunctiongradient::Function,
    p0::Vector,
    stepsize::Float64;
    whichstep = "EDStep",
    maxstepsize = 5,
    initialtime,
    maxseconds,
    homotopyMethod = "HomotopyContinuation",
    r = 1e-4,
    s = 0.9,
    traditional_newton = true,
)
    _, _, basegradient = get_NTv(p0, G, evaluateobjectivefunctiongradient)
    α = [0, stepsize]
    p = Base.copy(p0)
    while true
        try
            global q, success = stepchoice(
                G,
                whichstep,
                α[end],
                p0,
                basegradient;
                traditional_newton =  traditional_newton=traditional_newton ,
            )
        catch e
            @warn e
            maxstepsize = (α[1]+α[2])/2
            α = [α[1], (α[1]+α[2])/2]
            continue
        end
        _, Tq, vq = get_NTv(q, G, evaluateobjectivefunctiongradient)
        if time()-initialtime > maxseconds
            return q, Tq, vq, success, α[end]
        end
        if (
            (
                Q(q) > Q(p0) - r*α[end]*basegradient'*basegradient ||
                (Q(q) > Q(p0) && q!=p0)
            ) && success
        )
            helper = zoom(
                α[end-1],
                α[end],
                Q,
                evaluateobjectivefunctiongradient,
                G,
                whichstep,
                p0,
                basegradient,
                r,
                s;
                initialtime,
                maxseconds,
                homotopyMethod,
                traditional_newton,
            )
            _, Tq, vq = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
            return helper[1], Tq, vq, helper[2], helper[end]
        end
        if (abs(basegradient'*vq) <= s*abs(basegradient'*basegradient)) && success
            return q, Tq, vq, success, α[end]
        end
        if basegradient'*vq <= 0 && success
            helper = zoom(
                α[end],
                α[end-1],
                Q,
                evaluateobjectivefunctiongradient,
                G,
                whichstep,
                p0,
                basegradient,
                r,
                s;
                initialtime,
                maxseconds,
                homotopyMethod,
                traditional_newton,
            )
            _, Tq, vq = get_NTv(helper[1], G, evaluateobjectivefunctiongradient)
            return helper[1], Tq, vq, helper[2], helper[end]
        end
        if (success)
            push!(α, 2*α[end])
            p = q
        else
            _, Tp, vp = get_NTv(p, G, evaluateobjectivefunctiongradient)
            return p, Tp, vp, success, α[end]
        end
        deleteat!(α, 1)
        if α[end] > maxstepsize
            return q, Tq, vq, success, α[end-1]
        end
    end
end

#=
Zoom in on the step lengths between αlo and αhi to find the optimal step size here. This is part of the backtracking line search
=#
function zoom(
    αlo,
    αhi,
    Q,
    evaluateobjectivefunctiongradient,
    G,
    whichstep,
    p0,
    basegradient,
    r,
    s;
    initialtime,
    maxseconds,
    homotopyMethod,
    traditional_newton,
)
    qlo, suclo = stepchoice(
        G,
        whichstep,
        αlo,
        p0,
        basegradient;
        traditional_newton = traditional_newton,
    )
    # To not get stuck in the iteration, we use a for loop instead of a while loop
    # TODO Add a more meaningful stopping criterion
    index = 1
    while index <= 7
        global α = 0.5*(αlo+αhi)
        try
            global q, success = stepchoice(
                G,
                whichstep,
                α,
                p0,
                basegradient;
                traditional_newton = traditional_newton,
            )
        catch e
            @warn e
            αlo = αlo < αhi ? αlo : 0.5*(αlo+αhi)
            αhi = αlo < αhi ? 0.5*(αlo+αhi) : αhi
            continue
        end
        _, _, vq = get_NTv(q, G, evaluateobjectivefunctiongradient)
        if !success || time()-initialtime > maxseconds
            return q, success, α
        end

        if Q(q) > Q(p0) - r*α*basegradient'*basegradient
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
function get_NTv(
    q,
    G::SemialgebraicSet,
    evaluateobjectivefunctiongradient::Function;
    tol = 1e-8,
)
    active_indices = [
        i for (i, eq) in enumerate(G.fullequations) if
        i>length(G.equalities) && abs(evaluate(eq, G.variables=>q)) < tol
    ]#
    full_jacobian = evaluate.(G.fulljacobian, G.variables => q)
    active_jacobian = full_jacobian[:, vcat(1:length(G.equalities), active_indices)]
    ∇Qq = evaluateobjectivefunctiongradient(q)
    w = -∇Qq

    violated_indices = []
    for i in active_indices
        w'*full_jacobian[:, i] < tol ? push!(violated_indices, i) : nothing
    end

    semiactive_jacobian =
        full_jacobian[:, Vector{Int}(vcat(1:length(G.equalities), violated_indices))]

    try
        global Q_active = svd(Matrix{Float64}(active_jacobian)).U
        global Q_violated = svd(Matrix{Float64}(semiactive_jacobian)).U
    catch e
        @warn e
        global Q_active = qr(Matrix{Float64}(active_jacobian)).Q
        global Q_violated = qr(Matrix{Float64}(semiactive_jacobian)).Q
    end

    if length(G.equalities)==0 && length(active_indices)==0
        Nq_active, Tq_active = [0.0 for _ = 1:length(G.variables)],
        nullspace(zeros(Float64, length(G.variables), length(G.variables)))
    else
        Tq_active = nullspace(active_jacobian')
        Nq_active = Q_active[:, 1:(length(G.variables)-size(Tq_active)[2])] # O.N.B. for the normal space at q
    end

    if length(G.equalities)==0 && length(violated_indices)==0
        return Nq_active, Tq_active, -∇Qq
    end
    Tq_violated = nullspace(semiactive_jacobian')
    Nq_violated = Q_violated[:, 1:(length(G.variables)-size(Tq_violated)[2])] # O.N.B. for the normal (half-)space at q
    vq = w - Nq_violated * (Nq_violated' * w) # projected gradient -∇Q(p) onto the tangent cone, subtract the normal components
    @assert all(t->isapprox(t, 0, atol = tol), semiactive_jacobian'*vq)
    return Nq_active, Tq_active, vq
end

#=
 Parallel transport the vector vj from the tangent space Tj to the tangent space Ti
=#
function paralleltransport(vj, Tj, Ti)
    # transport vj ∈ Tj to become a vector ϕvj ∈ Ti
    # cols(Tj) give ONB for home tangent space, cols(Ti) give ONB for target tangent space
    U, _, Vt = svd(Ti' * Tj)
    Oij = U * Vt' # closest orthogonal matrix to the matrix (Ti' * Tj) comes from svd, remove \Sigma
    ϕvj = Ti * Oij * (Tj' * vj)
    return ϕvj
end

#= Take `maxsteps` steps to try and converge to an optimum. In each step, we use backtracking linesearch
to determine the optimal step size to go along the search direction
WARNING This is redundant and can be merged with findminima
=#
function takelocalsteps(
    p::Vector{Float64},
    ε0::Float64,
    tolerance,
    G::SemialgebraicSet,
    objectiveFunction::Function,
    evaluateobjectivefunctiongradient::Function;
    maxsteps = 6,
    maxstepsize = 4,
    decreasefactor = 2.5,
    initialtime = Base.time(),
    maxseconds = 100,
    whichstep = "EDStep",
    homotopyMethod = "HomotopyContinuation",
    momentum_strategy = true,
    tangent_predictor = [],
    θ_collect = [1],
    traditional_newton = true,
)
    _, Tp, vp = get_NTv(p, G, evaluateobjectivefunctiongradient)
    T_collect, timesturned = [Tp], 0 # normal spaces and tangent spaces, columns of Np and Tp are orthonormal bases
    q_collect, v_collect, n_collect = [p], [vp], [norm(vp)] # qs=new points on G, vs=projected gradients, ns=norms of projected gradients
    global stepsize = Base.copy(ε0)
    global tangent_predictor, stepsize_collect =
        isempty(tangent_predictor) ? Base.copy(p) : tangent_predictor, [stepsize]
    for _ = 1:maxsteps
        if !momentum_strategy
            global q, Tq, vq, success, stepsize = backtracking_linesearch(
                objectiveFunction,
                G,
                evaluateobjectivefunctiongradient,
                q_collect[end],
                Float64(stepsize);
                whichstep,
                maxstepsize,
                initialtime,
                maxseconds,
                homotopyMethod,
                traditional_newton = traditional_newton,
            )
            ϕvj = paralleltransport(vq, Tq, T_collect[end])
            if v_collect[end]'*ϕvj < 0
                timesturned += 1
                stepsize = stepsize/decreasefactor
            end
        else
            # "Topics in Convex Optimisation" §5, Cambridge, Hamza Fawzi for implementation details
            y = (1-θ_collect[end])*q_collect[end] + θ_collect[end]*tangent_predictor
            _q, success = stepchoice(
                G,
                whichstep,
                1,
                q_collect[end],
                y-q_collect[end];
                traditional_newton = traditional_newton,
            ) # Momentum Step
            if !success
                push!(θ_collect, θ_collect[end]/decreasefactor)
                continue
            end
            global q, Tq, vq, success, stepsize = backtracking_linesearch_momentum(
                objectiveFunction,
                G,
                evaluateobjectivefunctiongradient,
                _q,
                Float64(stepsize);
                whichstep,
                maxstepsize,
                initialtime,
                maxseconds,
                homotopyMethod,
                traditional_newton = traditional_newton,
            )
            push!(stepsize_collect, stepsize)
            tangent_predictor = q_collect[end]+(q-q_collect[end])/Base.abs(θ_collect[end])
            a = sqrt(θ_collect[end]^2*stepsize_collect[end]/stepsize_collect[end-1])
            ϕvj = paralleltransport(vq, Tq, T_collect[end])
            if v_collect[end]'*ϕvj < 0
                timesturned += 1
                push!(θ_collect, 0.85*(+a/2-sqrt(a^2+4)/2)/decreasefactor)
                stepsize = stepsize/decreasefactor
            else
                push!(θ_collect, 0.85*(-a/2+sqrt(a^2+4)/2))
            end
        end
        if Base.time() - initialtime > maxseconds
            break;
        end

        push!(q_collect, q);
        push!(T_collect, Tq);
        push!(n_collect, norm(vq, Inf));
        push!(v_collect, vq)
        length(T_collect)>3 ? deleteat!(T_collect, 1) : nothing
        if n_collect[end] < tolerance
            return q_collect[2:end],
            v_collect[2:end],
            n_collect[2:end],
            q,
            stepsize,
            true,
            timesturned,
            θ_collect,
            tangent_predictor
        end
        # The next (initial) stepsize is determined by the previous step and how much the energy function changed - in accordance with RieOpt.
        stepsize = Base.minimum([
            Base.maximum([
                success ?
                stepsize*(
                    (
                        v_collect[end-1]'*v_collect[end-1]
                    )*(
                        evaluateobjectivefunctiongradient(
                            q_collect[end-1],
                        )'*evaluateobjectivefunctiongradient(q_collect[end-1])
                    )
                )^(
                    1/4
                )/(
                    (
                        v_collect[end]'*v_collect[end]
                    )*(
                        evaluateobjectivefunctiongradient(
                            q_collect[end],
                        )'*evaluateobjectivefunctiongradient(q_collect[end])
                    )
                )^(1/4) : 0.1*stepsize,
                1e-4,
            ]),
            maxstepsize,
        ])
    end
    return q_collect[2:end],
    v_collect[2:end],
    n_collect[2:end],
    q_collect[end],
    stepsize,
    false,
    timesturned,
    θ_collect,
    tangent_predictor
end

#=
 Output object of the method `findminima`
=#
struct OptimizationResult
    is_minimization::Any
    computedpoints::Any
    initialstepsize::Any
    tolerance::Any
    converged::Any
    constraintvariety::Any
    objectivefunction::Any
    lastpointisoptimum::Any

    function Base.show(io::IO, opt::OptimizationResult)
        print("$(typeof(opt))(is_minimization: $(opt.is_minimization), computedpoints: $(opt.computedpoints), tolerance: $(opt.tolerance), converged: $(opt.converged), lastpointisoptimum: $(opt.lastpointisoptimum))")
    end

    function OptimizationResult(
        is_minimization,
        ps,
        ε0,
        tolerance,
        converged,
        G,
        Q,
        lastpointisoptimum,
    )
        new(is_minimization, ps, ε0, tolerance, converged, G, Q, lastpointisoptimum)
    end
end

#=
 The main function of this package. Given an initial point, a tolerance, an objective function and a constraint variety,
 we try to find the objective function's closest local minimum to the initial guess.
=#
function minimize(
    p0::Vector{Float64},
    tolerance::Float64,
    G::SemialgebraicSet,
    objectiveFunction::Function;
    maxseconds = 100,
    maxlocalsteps = 5,
    initialstepsize = 0.2,
    whichstep = "EDStep",
    initialtime = Base.time(),
    homotopyMethod = "HomotopyContinuation",
    momentum_strategy = true,
    momentum_factor = 0.9,
    traditional_newton = true,
)
    #TODO Rework minimality: We are not necessarily at a minimality, if resolveSingularity does not find any better point. => first setequations, then ismin
    global p = copy(p0) # initialize before updating `p` below
    ps = [p0] # record the *main steps* from p0, newp, newp, ... until converged
    evaluateobjectivefunctiongradient = x -> gradient(objectiveFunction, x)
    # initialize stepsize. Different to RieOpt! Logic: large projected gradient=>far away, large stepsize is admissible.
    global ε0 = initialstepsize
    global θ_collect, tangent_predictor = [Float64(momentum_factor)], Base.copy(p0)
    while (Base.time() - initialtime) <= maxseconds
        # update LSR, only store the *last local run*
        global computedpoints,
        _,
        _,
        q,
        suggestedstepsize,
        converged,
        _,
        θ_collect,
        tangent_predictor = takelocalsteps(
            p,
            ε0,
            tolerance,
            G,
            objectiveFunction,
            evaluateobjectivefunctiongradient;
            maxsteps = maxlocalsteps,
            initialtime = initialtime,
            maxseconds = maxseconds,
            whichstep = whichstep,
            homotopyMethod = homotopyMethod,
            momentum_strategy = momentum_strategy,
            tangent_predictor = tangent_predictor,
            θ_collect = θ_collect,
            traditional_newton = traditional_newton,
        )
        global ε0 = suggestedstepsize # update and try again!
        append!(ps, computedpoints)
        if converged
            # TODO detect singularities
            if norm(ps[end-1]-ps[end]) < tolerance^3
                optimality = isMinimum(
                    G,
                    objectiveFunction,
                    evaluateobjectivefunctiongradient,
                    ps[end];
                    traditional_newton = traditional_newton,
                    criticaltol = tolerance,
                )
                if optimality
                    return OptimizationResult(
                        true,
                        ps,
                        initialstepsize,
                        tolerance,
                        true,
                        G,
                        objectiveFunction,
                        optimality,
                    )
                end
                println("Resolving")
                _q, foundsomething = resolveSingularity(
                    q,
                    G,
                    objectiveFunction,
                    evaluateobjectivefunctiongradient;
                    homotopyMethod = homotopyMethod,
                    traditional_newton = traditional_newton,
                )
                if foundsomething
                    optRes = minimize(
                        _q,
                        tolerance,
                        G,
                        objectiveFunction;
                        maxseconds = maxseconds,
                        maxlocalsteps = maxlocalsteps,
                        initialstepsize = suggestedstepsize,
                        whichstep = whichstep,
                        initialtime = initialtime,
                        homotopyMethod = homotopyMethod,
                    )
                    return OptimizationResult(
                        true,
                        vcat(ps, optRes.computedpoints),
                        initialstepsize,
                        tolerance,
                        optRes.converged,
                        G,
                        objectiveFunction,
                        optRes.lastpointisoptimum,
                    )
                end
                return OptimizationResult(
                    true,
                    ps,
                    initialstepsize,
                    tolerance,
                    true,
                    G,
                    objectiveFunction,
                    optimality,
                )
            else
                optimality = isMinimum(
                    G,
                    objectiveFunction,
                    evaluateobjectivefunctiongradient,
                    ps[end];
                    traditional_newton = traditional_newton,
                    criticaltol = tolerance,
                )
                if !optimality
                    optRes = minimize(
                        q,
                        tolerance,
                        G,
                        objectiveFunction;
                        maxseconds = maxseconds,
                        maxlocalsteps = maxlocalsteps,
                        initialstepsize = suggestedstepsize,
                        whichstep = whichstep,
                        initialtime = initialtime,
                    )
                    return OptimizationResult(
                        true,
                        vcat(ps, optRes.computedpoints),
                        initialstepsize,
                        tolerance,
                        optRes.converged,
                        G,
                        objectiveFunction,
                        optRes.lastpointisoptimum,
                    )
                end
                return OptimizationResult(
                    true,
                    ps,
                    initialstepsize,
                    tolerance,
                    true,
                    G,
                    objectiveFunction,
                    optimality,
                )
            end
        else
            global p = q
        end
    end

    display(
        "We ran out of time... Try setting `maxseconds` to a larger value than $(maxseconds)",
    )
    optimality = isMinimum(
        G,
        objectiveFunction,
        evaluateobjectivefunctiongradient,
        ps[end];
        criticaltol = tolerance,
    )
    return OptimizationResult(
        true,
        ps,
        p0,
        ε0,
        tolerance,
        converged,
        G,
        evaluateobjectivefunctiongradient,
        optimality,
    )
end

function maximize(
    p0,
    tolerance,
    G::SemialgebraicSet,
    objectiveFunction::Function;
    kwargs...,
)
    optres = minimize(p0, tolerance, G, x->-objectiveFunction(x); kwargs...)
    return OptimizationResult(
        false,
        optres.computedpoints,
        p0,
        optres.initialstepsize,
        tolerance,
        optres.converged,
        optres.lastlocalstepresult,
        G,
        objectiveFunction,
        optres.lastpointisoptimum,
    )
end

# Below are functions `watch` and `draw`
# to visualize low-dimensional examples
function watch(
    result::OptimizationResult;
    totalseconds = 6.0,
    framesize = nothing,
    canvas_size = (800, 800),
    sampling_resolution = 100,
    kwargs...,
)
    if canvas_size[1] != canvas_size[2]
        @warn "Canvas is expected to be a square."
    end
    ps = result.computedpoints
    samples = result.constraintvariety.samples
    if !isempty(samples)
        mediannorm = (sort([norm(p) for p in samples]))[Int(floor(samples/2))]
        samples = filter(x -> norm(x) < 2*mediannorm+0.5, samples)
    end
    framespersecond = length(ps) / totalseconds
    if framespersecond > 45
        framespersecond = 45
    end
    startingtime = Base.time()
    dim = length(ps[1])
    anim = Animation()
    if dim == 2
        if framesize==nothing
            fullx = [
                minimum([q[1] for q in vcat(samples, ps)]) - 0.025,
                maximum([q[1] for q in vcat(samples, ps)]) + 0.025,
            ]
            fully = [
                minimum([q[2] for q in vcat(samples, ps)]) - 0.025,
                maximum([q[2] for q in vcat(samples, ps)]) + 0.025,
            ]
        else
            if !(framesize isa Union{Vector,Tuple}) ||
               length(framesize)!=2 ||
               any(fr->fr[2]-fr[1]<1e-4, framesize)
                throw(
                    error(
                        "Framesize needs to be a tuple of (nonempty) intervals, but is $(typeof(framesize)). Use for example `framesize=((-1.5,1.5),(-1.5,1.5))`.",
                    ),
                )
            end
            fullx = framesize[1]
            fully = framesize[2]
        end
        initplt = plot(
            [],
            [],
            xlims = fullx,
            ylims = fully,
            left_margin = 16mm,
            legend = false,
            size = canvas_size,
            tickfontsize = 16*canvas_size[1]/800,
            grid = false,
        )

        x_array, y_array = [
            fullx[1]+i*(fullx[2]-fullx[1])/sampling_resolution for
            i = 0:sampling_resolution
        ],
        [fully[1]+j*(fully[2]-fully[1])/sampling_resolution for j = 0:sampling_resolution]
        heatmap_array = [
            [x_array[i+1], y_array[j+1]] for i = 0:sampling_resolution for
            j = 0:sampling_resolution
        ]
        for eq in result.constraintvariety.inequalities
            heatmap_array = filter(
                point->evaluate(eq, result.constraintvariety.variables=>point) >= 0,
                heatmap_array,
            )
        end
        scatter!(
            initplt,
            [ar[1] for ar in heatmap_array],
            [ar[2] for ar in heatmap_array],
            markershape = :rect,
            markersize = 4*(100/sampling_resolution)*(canvas_size[1]/800),
            markerstrokewidth = 0,
            color = RGBA{Float64}(0.75, 0.75, 0.75),
        )

        for eq in result.constraintvariety.equalities
            implicit_plot!(initplt, x->evaluate(eq, result.constraintvariety.variables=>x))
        end

        if result.lastpointisoptimum
            initplt = scatter!(
                initplt,
                [ps[end][1]],
                [ps[end][2]],
                legend = false,
                markersize = 17.5,
                color = :red,
                markershape = :rtriangle,
                xlims = fullx,
                ylims = fully,
            )
        end

        frame(anim)
        for p in ps[1:end]
            # BELOW: only plot next point, delete older points during animation
            # plt = scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            # BELOW: keep old points during animation.
            initplt = scatter!(
                initplt,
                [p[1]],
                [p[2]],
                legend = false,
                markersize = 7.5,
                alpha = 0.75,
                color = :black,
                xlims = fullx,
                ylims = fully,
            )
            frame(anim)
        end
        return gif(anim, "optimization_animation_$startingtime.gif", fps = framespersecond)
    else
        throw(error("The only currently supported dimension is 2."))
    end
end

function draw(
    result::OptimizationResult;
    framesize = nothing,
    canvas_size = (800, 800),
    sampling_resolution = 100,
    kwargs...,
)
    if canvas_size[1] != canvas_size[2]
        @warn "Canvas is expected to be a square."
    end
    dim = length(result.computedpoints[1]) # dimension of the ambient space
    ps = result.computedpoints
    samples = result.constraintvariety.samples
    if !isempty(samples)
        mediannorm = (sort([norm(p) for p in samples]))[Int(floor(samples/2))]
        samples = filter(x -> norm(x) < 2*mediannorm+0.5, samples)
    end
    startingtime = Base.time()
    if dim == 2
        if framesize==nothing
            fullx = [
                minimum([q[1] for q in vcat(samples, ps)]) - 0.025,
                maximum([q[1] for q in vcat(samples, ps)]) + 0.025,
            ]
            fully = [
                minimum([q[2] for q in vcat(samples, ps)]) - 0.025,
                maximum([q[2] for q in vcat(samples, ps)]) + 0.025,
            ]
        else
            if !(framesize isa Union{Vector,Tuple}) ||
               length(framesize)!=2 ||
               any(fr->fr[2]-fr[1]<1e-4, framesize)
                throw(
                    error(
                        "Framesize needs to be a tuple of (nonempty) intervals, but is $(typeof(framesize)). Use for example `framesize=((-1.5,1.5),(-1.5,1.5))`.",
                    ),
                )
            end
            fullx = framesize[1]
            fully = framesize[2]
        end
        initplt = plot(
            [],
            [],
            xlims = fullx,
            ylims = fully,
            left_margin = 16mm,
            legend = false,
            size = canvas_size,
            title = "Optimization Points",
            titlefontsize = 20*canvas_size[1]/800,
            tickfontsize = 18*canvas_size[1]/800,
        )
        x_array, y_array = [
            fullx[1]+i*(fullx[2]-fullx[1])/sampling_resolution for
            i = 0:sampling_resolution
        ],
        [fully[1]+j*(fully[2]-fully[1])/sampling_resolution for j = 0:sampling_resolution]
        heatmap_array = [
            [x_array[i+1], y_array[j+1]] for i = 0:sampling_resolution for
            j = 0:sampling_resolution
        ]
        for eq in result.constraintvariety.inequalities
            heatmap_array = filter(
                point->evaluate(eq, result.constraintvariety.variables=>point) >= 0,
                heatmap_array,
            )
        end
        scatter!(
            initplt,
            [ar[1] for ar in heatmap_array],
            [ar[2] for ar in heatmap_array],
            markershape = :rect,
            markersize = 4*(100/sampling_resolution)*(canvas_size[1]/800),
            markerstrokewidth = 0,
            color = RGBA{Float64}(0.75, 0.75, 0.75),
        )
        for eq in result.constraintvariety.equalities
            implicit_plot!(initplt, x->evaluate(eq, result.constraintvariety.variables=>x))
        end
        for q in ps
            scatter!(
                initplt,
                [q[1]],
                [q[2]],
                legend = false,
                color = :black,
                xlims = fullx,
                ylims = fully,
            )
        end

        localqs = ps[Int(ceil(length(ps)/2)):end]
        zoomx = [
            minimum([q[1] for q in localqs]) - 0.025,
            maximum([q[1] for q in localqs]) + 0.025,
        ]
        zoomy = [
            minimum([q[2] for q in localqs]) - 0.025,
            maximum([q[2] for q in localqs]) + 0.025,
        ]
        initplt2 = plot(
            [],
            [],
            xlims = zoomx,
            ylims = zoomy,
            left_margin = 16mm,
            legend = false,
            size = canvas_size,
            title = "Zoomed-In Optimization Points",
            titlefontsize = 20*canvas_size[1]/800,
            tickfontsize = 18*canvas_size[1]/800,
        )
        x_array, y_array = [
            zoomx[1]+i*(zoomx[2]-zoomx[1])/sampling_resolution for
            i = 0:sampling_resolution
        ],
        [zoomy[1]+j*(zoomy[2]-zoomy[1])/sampling_resolution for j = 0:sampling_resolution]
        heatmap_array = [
            [x_array[i+1], y_array[j+1]] for i = 0:sampling_resolution for
            j = 0:sampling_resolution
        ]
        for eq in result.constraintvariety.inequalities
            heatmap_array = filter(
                point->evaluate(eq, result.constraintvariety.variables=>point) >= 0,
                heatmap_array,
            )
        end
        scatter!(
            initplt2,
            [ar[1] for ar in heatmap_array],
            [ar[2] for ar in heatmap_array],
            markershape = :rect,
            markersize = 4*(100/sampling_resolution)*(canvas_size[1]/800),
            markerstrokewidth = 0,
            color = RGBA{Float64}(0.75, 0.75, 0.75),
        )
        for eq in result.constraintvariety.equalities
            implicit_plot!(initplt2, x->evaluate(eq, result.constraintvariety.variables=>x))
        end
        for q in localqs
            scatter!(
                initplt2,
                [q[1]],
                [q[2]],
                legend = false,
                color = :blue,
                xlims = zoomx,
                ylims = zoomy,
            )
        end

        energy_values = [result.objectivefunction(p) for p in ps]
        pltvnorms = plot(
            1:length(energy_values),
            energy_values,
            left_margin = 16mm,
            size = canvas_size,
            legend = false,
            title = "Objective Function Values",
            titlefontsize = 20*canvas_size[1]/800,
            tickfontsize = 18*canvas_size[1]/800,
        )
        plt = plot(
            initplt,
            initplt2,
            pltvnorms,
            layout = (3, 1),
            size = (canvas_size[1]+100, canvas_size[2]*3+100),
        )
        return plt
    else
        throw(error("The only currently supported dimension is 2."))
    end
end

end
