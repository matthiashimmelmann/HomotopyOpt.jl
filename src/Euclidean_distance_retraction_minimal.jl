module Euclidean_distance_retraction_minimal

import HomotopyContinuation:
    results,
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
    track
import LinearAlgebra:
    norm,
    transpose,
    qr,
    rank,
    normalize,
    pinv,
    eigvals,
    abs,
    eigvecs,
    svd,
    nullspace,
    I,
    Symmetric,
    cholesky
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
    gaussnewtonstep_HC,
    EDStep,
    EDStep_HC,
    #INFO: The following package is not maintained by us. Find it here: https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl
    HomotopyContinuation

#=
 Equips a HomotopyContinuation.Tracker with a start Solution that can be changed on the fly
=#

mutable struct TrackerWithStartSolution
    tracker::Any
    startSolution::Any
    jacobian::Any
    variables::Any
    parameters::Any

    function TrackerWithStartSolution(T::Tracker, startSol::Vector, d::Int)
        @var t point[1:d] vector[1:d]
        jacobian_z = Symmetric(
            hcat(
                [
                    differentiate(eq, T.homotopy.F.interpreted.system.variables) for
                    eq in T.homotopy.F.interpreted.system.expressions
                ]...,
            ),
        )
        new(
            T,
            startSol,
            jacobian_z,
            T.homotopy.F.interpreted.system.variables,
            T.homotopy.F.interpreted.system.parameters,
        )
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
    variables::Any
    equations::Any
    fullequations::Any
    jacobian::Any
    ambientdimension::Any
    dimensionofvariety::Any
    EDTracker::Any

    # Given variables and HomotopyContinuation-based equations, sample points from the variety and return the corresponding struct
    function ConstraintVariety(varz, eqnz, N::Int, d::Int)
        jacobian = hcat([differentiate(eq, varz) for eq in eqnz]...)

        fulleqnz = eqnz
        if length(eqnz) + d > N
            eqnz = randn(Float64, N-d, length(eqnz))*eqnz
        end

        @var u[1:N]
        @var λ[1:length(eqnz)]
        Lagrange = 0.5*sum((varz-u) .^ 2) + sum(λ .* eqnz)
        ∇Lagrange = differentiate(Lagrange, vcat(varz, λ))
        EDSystem = System(∇Lagrange, variables = vcat(varz, λ), parameters = u)
        p0 = randn(Float64, N)
        H = ParameterHomotopy(EDSystem, start_parameters = p0, target_parameters = p0)
        EDTracker = TrackerWithStartSolution(Tracker(H), [], N)
        new(varz, eqnz, fulleqnz, jacobian, N, d, EDTracker)
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
    function ConstraintVariety(eqnz, N::Int, d::Int)
        ConstraintVariety(eqnz::Function, N::Int, d::Int)
    end

    #Let the dimension be determined by the algorithm and calculate samples
    function ConstraintVariety(varz, eqnz, p::Vector{Float64})
        G = ConstraintVariety(varz, eqnz, length(varz), 0)
        setEquationsAtp!(G, p)
        return (G)
    end

    #Only let the dimension be determined by the algorithm
    function ConstraintVariety(varz, eqnz, p::Vector{Float64})
        G = ConstraintVariety(varz, eqnz, length(varz), 0)
        setEquationsAtp!(G, p)
        return (G)
    end
end


#=
Add Samples to an already existing ConstraintVariety
=#
function setEquationsAtp!(G::ConstraintVariety, p; tol = 1e-5)
    jacobianRank = rank(evaluate.(G.jacobian, G.variables=>p); atol = tol)
    eqnz = G.fullequations
    if length(eqnz) + (G.ambientdimension-jacobianRank) > G.ambientdimension
        eqnz = randn(Float64, jacobianRank, length(eqnz))*eqnz
    end
    setfield!(G, :equations, eqnz)
    setfield!(G, :dimensionofvariety, (G.ambientdimension-jacobianRank))
end

function gaussnewtonstep_HC(G::ConstraintVariety, initial_point, q; max_iters)
    res = HomotopyContinuation.newton(
        G.EDTracker.tracker.homotopy.F,
        initial_point,
        q;
        max_iters = max_iters,
    )
    #display(res)
    return real.(res.x), res.iters
end
#=
 We predict in the projected gradient direction and correct by using the Gauss-Newton method
=#
function gaussnewtonstep(
    equations,
    jacobian,
    vars,
    p;
    tol = 1e-10,
    initialtime = Base.time(),
    maxtime = 50,
    maxsteps = 2,
    factor = 1,
)
    global q = p
    global iter = 0
    while norm(evaluate.(equations, vars=>q)) > tol && iter <= maxsteps
        if Base.time()-initialtime > maxtime
            break
        end
        J = Matrix{Float64}(evaluate.(jacobian, vars=>q))
        global q = q .- (factor .* J') \ evaluate.(equations, vars=>q)
        global iter += 1
    end
    return q, iter
end

function EDStep_HC(
    G::ConstraintVariety,
    p,
    v;
    homotopyMethod,
    euler_step = "explicit",
    amount_Euler_steps = 0,
    maxtime = 100,
    print = false,
)
    #initialtime = Base.time()
    q0 = p#+1e-3*Basenormal[:,1]
    start_parameters!(G.EDTracker.tracker, q0)
    global linear_solves = 0
    setStartSolution(G.EDTracker, vcat(p, [0.0 for _ in G.equations]))

    if homotopyMethod=="HomotopyContinuation"
        q = p+v
        target_parameters!(G.EDTracker.tracker, q)
        tracker = track(G.EDTracker.tracker, G.EDTracker.startSolution)
        result = solution(tracker)
        if print
            display(tracker)
            display(result)
        end
        if all(entry->Base.abs(entry.im)<1e-4, result)
            return [entry.re for entry in result[1:length(p)]],
            tracker.accepted_steps,
            tracker
        else
            throw(error("Complex Space entered!"))
        end
    else
        global currentSolution = G.EDTracker.startSolution
        if amount_Euler_steps!=-1
            q = p+(1/(amount_Euler_steps+1))*v
            #equations = evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
            if euler_step=="explicit"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        0,
                        1/(amount_Euler_steps+1);
                        trivial = true,
                    )
                #elseif euler_step=="newton"
                #    global currentSolution = currentSolution .+ vcat(v*(1/(amount_Euler_steps+1)), [0. for _ in 1:(length(currentSolution)-length(v))])
            elseif euler_step=="RK2"
                global currentSolution =
                    currentSolution .+ RK2step(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        0,
                        1/(amount_Euler_steps+1);
                        trivial = true,
                    )
                global linear_solves = linear_solves+1
            elseif euler_step=="implicit"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        0,
                        1/(amount_Euler_steps+1);
                        trivial = true,
                    )
            elseif euler_step=="heun"
                global currentSolution =
                    currentSolution .+
                    1/2*(
                        explicitEulerStep(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            0,
                            1/(amount_Euler_steps+1);
                            trivial = true,
                        )+implicitEulerStep(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            0,
                            1/(amount_Euler_steps+1),
                        )
                    )
            end
        else
            q = p+v
            #equations = evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
        end
        global currentSolution, gaussnewtonsolves = gaussnewtonstep_HC(
            G,
            currentSolution,
            q;
            max_iters = amount_Euler_steps<=0 ? 400 : (euler_step=="newton" ? 10 : 9),
        )
        global linear_solves = linear_solves + gaussnewtonsolves
        for step = 1:amount_Euler_steps
            q = p+((step+1)/(amount_Euler_steps+1))*v
            #prev_sol = currentSolution
            if euler_step=="explicit"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        step/(amount_Euler_steps+1),
                        1/(amount_Euler_steps+1);
                        trivial = false,
                    )
                global linear_solves = linear_solves+1
            elseif euler_step=="RK2"
                global currentSolution =
                    currentSolution .+ RK2step(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        step/(amount_Euler_steps+1),
                        1/(amount_Euler_steps+1);
                        trivial = false,
                    )
                global linear_solves = linear_solves+2
            elseif euler_step=="midpoint"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        step/(amount_Euler_steps+1),
                        1/(2*(amount_Euler_steps+1));
                        trivial = false,
                    )
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        1/(2*(amount_Euler_steps+1)) + step/(amount_Euler_steps+1),
                        1/(2*(amount_Euler_steps+1));
                        trivial = false,
                    )
                global linear_solves = linear_solves+2
            elseif euler_step=="heun"
                global currentSolution =
                    currentSolution .+
                    1/2*(
                        explicitEulerStep(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            step/(amount_Euler_steps+1),
                            1/(amount_Euler_steps+1);
                            trivial = false,
                        ) .+ RK2step(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            step/(amount_Euler_steps+1),
                            1/(amount_Euler_steps+1),
                        )
                    )
                global linear_solves = linear_solves+2
            end
            #equations = evaluate(G.EDTracker.tracker.homotopy.F.interpreted.system.expressions, G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q)
            global currentSolution, gaussnewtonsolves = gaussnewtonstep_HC(
                G,
                currentSolution,
                q;
                max_iters = amount_Euler_steps==step ? 400 :
                            (euler_step=="newton" ? 10 : 9),
            )
            global linear_solves = linear_solves + gaussnewtonsolves
        end
        #println(norm(prev_sol-currentSolution), " ", norm(prediction-currentSolution))
        return currentSolution[1:length(q)], linear_solves, []
    end
end


function EDStep(
    G::ConstraintVariety,
    p,
    v;
    homotopyMethod,
    tol = 1e-10,
    euler_step = "explicit",
    amount_Euler_steps = 0,
    maxtime = 100,
)
    initialtime = Base.time()
    q0 = p#+1e-3*Basenormal[:,1]
    start_parameters!(G.EDTracker.tracker, q0)
    global linear_solves = 0
    setStartSolution(G.EDTracker, vcat(p, [0.0 for _ in G.equations]))

    if homotopyMethod=="HomotopyContinuation"
        q = p+v
        target_parameters!(G.EDTracker.tracker, q)
        tracker = track(G.EDTracker.tracker, G.EDTracker.startSolution)
        result = solution(tracker)
        if all(entry->Base.abs(entry.im)<1e-4, result)
            return [entry.re for entry in result[1:length(p)]], tracker.accepted_steps
        else
            throw(error("Complex Space entered!"))
        end
    else
        currentSolution = G.EDTracker.startSolution
        vars = G.EDTracker.tracker.homotopy.F.interpreted.system.variables
        if amount_Euler_steps!=-1
            q = p+(1/(amount_Euler_steps+1))*v
            equations = evaluate(
                G.EDTracker.tracker.homotopy.F.interpreted.system.expressions,
                G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q,
            )
            if euler_step=="explicit"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        0,
                        1/(amount_Euler_steps+1);
                        trivial = true,
                    )
            elseif euler_step=="RK2"
                global currentSolution =
                    currentSolution .+ RK2step(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        0,
                        1/(amount_Euler_steps+1);
                        trivial = true,
                    )
                global linear_solves = linear_solves+1
            elseif euler_step=="implicit"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        0,
                        1/(amount_Euler_steps+1);
                        trivial = true,
                    )
            elseif euler_step=="heun"
                global currentSolution =
                    currentSolution .+
                    1/2*(
                        explicitEulerStep(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            0,
                            1/(amount_Euler_steps+1);
                            trivial = true,
                        )+implicitEulerStep(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            0,
                            1/(amount_Euler_steps+1),
                        )
                    )
            end
        else
            q = p+v
            equations = evaluate(
                G.EDTracker.tracker.homotopy.F.interpreted.system.expressions,
                G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q,
            )
        end
        global currentSolution, gaussnewtonsolves = gaussnewtonstep(
            equations,
            G.EDTracker.jacobian,
            vars,
            currentSolution;
            initialtime,
            maxtime,
            maxsteps = amount_Euler_steps<=0 ? 250 : (euler_step=="newton" ? 3 : 2),
        )
        global linear_solves = linear_solves + gaussnewtonsolves
        for step = 1:amount_Euler_steps
            q = p+((step+1)/(amount_Euler_steps+1))*v
            #prev_sol = currentSolution
            if euler_step=="explicit"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        step/(amount_Euler_steps+1),
                        1/(amount_Euler_steps+1);
                        trivial = false,
                    )
                global linear_solves = linear_solves+1
            elseif euler_step=="RK2"
                global currentSolution =
                    currentSolution .+ RK2step(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        step/(amount_Euler_steps+1),
                        1/(amount_Euler_steps+1);
                        trivial = false,
                    )
                global linear_solves = linear_solves+2
            elseif euler_step=="midpoint"
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        step/(amount_Euler_steps+1),
                        1/(2*(amount_Euler_steps+1));
                        trivial = false,
                    )
                global currentSolution =
                    currentSolution .+ explicitEulerStep(
                        G.EDTracker,
                        currentSolution,
                        p,
                        v,
                        1/(2*(amount_Euler_steps+1)) + step/(amount_Euler_steps+1),
                        1/(2*(amount_Euler_steps+1));
                        trivial = false,
                    )
                global linear_solves = linear_solves+2
            elseif euler_step=="heun"
                global currentSolution =
                    currentSolution .+
                    1/2*(
                        explicitEulerStep(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            step/(amount_Euler_steps+1),
                            1/(amount_Euler_steps+1);
                            trivial = false,
                        ) .+ RK2step(
                            G.EDTracker,
                            currentSolution,
                            p,
                            v,
                            step/(amount_Euler_steps+1),
                            1/(amount_Euler_steps+1),
                        )
                    )
                global linear_solves = linear_solves+2
            end
            equations = evaluate(
                G.EDTracker.tracker.homotopy.F.interpreted.system.expressions,
                G.EDTracker.tracker.homotopy.F.interpreted.system.parameters => q,
            )
            global currentSolution, gaussnewtonsolves = gaussnewtonstep(
                equations,
                G.EDTracker.jacobian,
                vars,
                currentSolution;
                initialtime,
                maxtime,
                maxsteps = amount_Euler_steps==step ? 250 : (euler_step=="newton" ? 3 : 2),
            )
            global linear_solves = linear_solves + gaussnewtonsolves
        end
        #println(norm(prev_sol-currentSolution), " ", norm(prediction-currentSolution))
        return currentSolution[1:length(q)], linear_solves
    end
end

function explicitEulerStep(
    EDTracker::TrackerWithStartSolution,
    q,
    p,
    v,
    prev_step,
    step_size;
    trivial = false,
)
    if trivial
        return vcat(v*step_size, [0.0 for _ = 1:(length(q)-length(p))])
    end
    dz = evaluate.(
        EDTracker.jacobian,
        vcat(EDTracker.variables, EDTracker.parameters) => vcat(q, p+prev_step*v),
    )
    du = vcat(v*step_size, [0.0 for _ = 1:(length(q)-length(p))])
    return dz \ (du)
end

function RK2step(
    EDTracker::TrackerWithStartSolution,
    q,
    p,
    v,
    prev_step,
    step_size;
    trivial = false,
)
    if trivial
        k1 = vcat(
            0.5*v*step_size,
            [
                0.0 for _ =
                1:(length(
                    EDTracker.tracker.homotopy.F.interpreted.system.variables,
                )-length(p))
            ],
        )
    else
        dz1 =
            -evaluate.(
                EDTracker.jacobian,
                vcat(EDTracker.variables, EDTracker.parameters) => vcat(q, p+prev_step*v),
            )
        du1 = evaluate.(
            EDTracker.jacobian_parameter,
            vcat(EDTracker.variables, EDTracker.ptv) => vcat(q, v),
        )
        k1 = dz1 \ (du1*0.5*step_size)
    end
    dz2 =
        -evaluate.(
            EDTracker.jacobian,
            vcat(
                EDTracker.tracker.homotopy.F.interpreted.system.variables,
                EDTracker.tracker.homotopy.F.interpreted.system.parameters,
            ) => vcat(q .+ k1, p+(0.5*step_size+prev_step)*v),
        )
    du2 = evaluate.(
        EDTracker.jacobian_parameter,
        vcat(EDTracker.tracker.homotopy.F.interpreted.system.variables, EDTracker.ptv) => vcat(q .+ k1, v),
    )
    return dz2 \ (du2*step_size)
end


#=
 Get the tangent and normal space of a ConstraintVariety at a point q
=#
function get_NTv(q, G::ConstraintVariety, v)
    dgq = evaluate.(G.jacobian, G.variables => q)
    Qq = svd(Matrix{Float64}(dgq)).U
    #index = count(p->p>1e-8, S)
    Nq = Qq[:, 1:(G.ambientdimension-G.dimensionofvariety)] # O.N.B. for the normal space at q
    Tq = nullspace(dgq')#(Qq.V)[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # O.N.B. for tangent space at q
    # we evaluate the gradient of the obj fcn at the point `q`
    ∇Qq1 = v
    w1 = -∇Qq1 # direction of decreasing energy function

    vq1 = w1 - Nq * (Nq' * w1) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components
    return Nq, Tq, vq1
end

end
