using HomotopyContinuation, LinearAlgebra, ImplicitPlots, Plots, Statistics, Dates

# this code modifies `ConstrainedOptimizationByParameterHomotopy.jl`
# so that instead of an ObjectiveFunction, you specify a function called `evaluateobjectivefunctiongradient`
# this makes more sense. You have to define this function yourself,
# but now it does not depend on symbolic algebra from HomotopyContinuation.jl

struct ConstraintVariety
    variables
    equations
    jacobian
    ambientdimension
    dimensionofvariety
    samples

    function ConstraintVariety(varz, eqnz, N::Int, d::Int, numsamples::Int)
        dg = HomotopyContinuation.differentiate(eqnz, varz)
        randL = HomotopyContinuation.rand_subspace(N; codim=d)
        randResult = HomotopyContinuation.solve(eqnz; target_subspace = randL, variables=varz)
        Ωs = []
        for _ in 1:numsamples
            newΩs = solve(
                    eqnz,
                    HomotopyContinuation.solutions(randResult);
                    variables = varz,
                    start_subspace = randL,
                    target_subspace = HomotopyContinuation.rand_subspace(N; codim = d, real = true),
                    transform_result = (R,p) -> HomotopyContinuation.real_solutions(R),
                    flatten = true
            )
            realsols = real_solutions(newΩs)
            push!(Ωs, realsols...)
        end
        new(varz,eqnz,dg,N,d,Ωs)
    end

    function ConstraintVariety(varz,eqnz,N::Int,d::Int)
        dg = HomotopyContinuation.differentiate(eqnz, varz)
        new(varz,eqnz,dg,N,d,[])
    end
end

function computesystem(p, G::ConstraintVariety,
                evaluateobjectivefunctiongradient::Function)

    dgp = HomotopyContinuation.evaluate(G.jacobian, G.variables => p)
    Up,_ = qr( transpose(dgp) )
    Np = Up[:, 1:(G.ambientdimension - G.dimensionofvariety)] # gives ONB for N_p(G) normal space

    # we evaluate the gradient of the obj fcn at the point `p`
    ∇Qp = evaluateobjectivefunctiongradient(p)

    w = -∇Qp # direction of decreasing energy function
    v = w - Np * (Np' * w) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components

    g = G.equations

    if G.dimensionofvariety > 1 # Need more linear equations when tangent space has dim > 1
        L,_ = LinearAlgebra.qr( hcat(v, Np))
        L = L[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # basis of the orthogonal complement of v inside T_p(G)
        L = L' * G.variables - L' * p # affine linear equations through p, containing v, give curve in variety along v
        u = LinearAlgebra.normalize(v)
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

function twostep(F, p, stepsize)
    # we want parameter homotopy from 0.0 to stepsize, so we take two steps
    # first from 0.0 to a complex number parameter, then from that parameter to stepsize.
    midparam = stepsize/2 + stepsize/2*1.0im # complex number *midway* between 0 and stepsize, but off real line
    solveresult = HomotopyContinuation.solve(F, [p]; start_parameters=[0.0 + 0.0im], target_parameters=[midparam])
    midsols = HomotopyContinuation.solutions(solveresult)
    success = false
    if length(midsols) > 0
        midsolution = midsols[1] # only tracked one solution path, thus there should only be one solution
        solveresult = HomotopyContinuation.solve(F, [midsolution]; start_parameters=[midparam],
                                                    target_parameters=[stepsize + 0.0im])
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

function getNandTandv(q, G::ConstraintVariety,
                    evaluateobjectivefunctiongradient::Function)

    dgq = evaluate(G.jacobian, G.variables => q)
    Qq,_ = LinearAlgebra.qr(transpose(dgq))
    Nq = Qq[:, 1:(G.ambientdimension - G.dimensionofvariety)] # O.N.B. for the normal space at q
    Tq = Qq[:, (G.ambientdimension - G.dimensionofvariety + 1):end] # O.N.B. for tangent space at q

    # we evaluate the gradient of the obj fcn at the point `q`
    ∇Qq = evaluateobjectivefunctiongradient(q)

    w = -∇Qq # direction of decreasing energy function
    vq = w - Nq * (Nq' * w) # projected gradient -∇Q(p) onto the tangent space, subtract the normal components

    return Nq, Tq, vq
end

function paralleltransport(vj, Tj, Ti)
    # transport vj ∈ Tj to become a vector ϕvj ∈ Ti
    # cols(Tj) give ONB for home tangent space, cols(Ti) give ONB for target tangent space
    U,_,Vt = LinearAlgebra.svd( Ti' * Tj )
    Oij = U * Vt # closest orthogonal matrix to the matrix (Ti' * Tj) comes from svd, remove \Sigma
    ϕvj = Ti * Oij * (Tj' * vj)
    return ϕvj
end

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

function takelocalsteps(p, ε0, tolerance, G::ConstraintVariety,
                evaluateobjectivefunctiongradient::Function;
                decreasefactor=5.0, increasefactor=1.1, maxsteps=50)

    keepgoing, converged, j, timesturned, valleysfound, count = true, false, 1, 0, 0, 0
    Np, Tp, vp = getNandTandv(p, G, evaluateobjectivefunctiongradient)
    Ns, Ts = [Np], [Tp] # normal spaces and tangent spaces, columns of Np and Tp are orthonormal bases
    qs, vs, ns = [p], [vp], [LinearAlgebra.norm(vp)] # qs=new points on G, vs=projected gradients, ns=norms of projected gradients
    F = computesystem(p, G, evaluateobjectivefunctiongradient) # sets up the system of equations, one parameter ε
    while keepgoing
        count += 1
        if count > maxsteps
            keepgoing = false
        end
        j = j + 1 # increase count for j ∈ 2,3,4,5,...  start j=2 since qs[1] = p.
        # For j ∈ 2,3,4,... we will call `twostep(F, p, j*ε0)`
        stepsize = (j-1) * ε0 # because p = qs[1] we are off by one here.
        q, success = twostep(F, p, stepsize)
        if success
            push!(qs, q)
            # now compute normal space, tangent spaces, projected gradient at the point q
            Nq, Tq, vq = getNandTandv(q, G, evaluateobjectivefunctiongradient)
            push!(Ns, Nq)
            push!(Ts, Tq)
            push!(vs, vq)
            push!(ns, norm(vq))
            if norm(vq) < tolerance
                keepgoing = false
                converged = true
                newp = q
                newε0 = ε0
                return LocalStepsResult(p,ε0,qs,vs,ns,newp,newε0,converged,timesturned,valleysfound)
            elseif ((ns[j] - ns[j-1]) > 0.0)
                if j > 2 && ((ns[j-1] - ns[j-2]) < 0.0)
                    # projected norms were decreasing, but started increasing!
                    # check parallel transport dot product to see if we should slow down
                    valleysfound += 1
                    ϕvj = paralleltransport(vs[j], Ts[j], Ts[j-2])
                    if ((vs[j-2]' * ϕvj) < 0.0)
                        # we think there is a critical point we skipped past! slow down!
                        timesturned += 1
                        newp = qs[j-2]
                        newε0 = ε0 / decreasefactor
                        return LocalStepsResult(p,ε0,qs,vs,ns,newp,newε0,converged,timesturned,valleysfound)
                    end
                end
            end
        else
            # our twostep homotopy failed, maybe we left the real-valued variety!
            keepgoing = false
        end
    end
    newp = qs[end] # is this the best choice?
    newε0 = ε0 * increasefactor
    return LocalStepsResult(p,ε0,qs,vs,ns,newp,newε0,converged,timesturned,valleysfound)
end

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

function watch(result::OptimizationResult; totalseconds=5.0)
    ps = result.computedpoints
    samples = result.constraintvariety.samples
    mediannorm = Statistics.median([LinearAlgebra.norm(p) for p in samples])
    samples = filter(x -> LinearAlgebra.norm(x) < 2*mediannorm, samples)
    initplt = plot() # initialize
    M = length(ps)
    framespersecond = M / totalseconds
    if framespersecond > 45
        framespersecond = 45
    end
    startingtime = Dates.now() #Base.time()
    dim = length(ps[1])
    anim = Animation()
    if dim == 2
        fullx = [minimum([q[1] for q in samples]) - 0.01, maximum([q[1] for q in samples]) + 0.01]
        fully = [minimum([q[2] for q in samples]) - 0.01, maximum([q[2] for q in samples]) + 0.01]
        g1 = result.constraintvariety.equations[1] # should only be a curve in ambient R^2
        initplt = implicit_plot(g1, xlims=fullx, ylims=fully, legend=false)
        frame(anim)
        for p in ps
            # BELOW: only plot next point, delete older points during animation
            # plt = scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            # BELOW: keep old points during animation.
            initplt = scatter!(initplt, [p[1]], [p[2]], legend=false, color=:black, xlims=fullx, ylims=fully)
            frame(anim)
        end
    elseif dim == 3
        fullx = [minimum([q[1] for q in samples]) - 0.01, maximum([q[1] for q in samples]) + 0.01]
        fully = [minimum([q[2] for q in samples]) - 0.01, maximum([q[2] for q in samples]) + 0.01]
        fullz = [minimum([q[3] for q in samples]) - 0.01, maximum([q[3] for q in samples]) + 0.01]
        initplt = plot() # initialize
        for q in samples
            initplt = scatter!(initplt, [q[1]], [q[2]], [q[3]],
                                    legend=false, color=:orange, markersize=3.0,
                                    xlims=fullx, ylims=fully, zlims=fullz)
        end
        frame(anim)
        for p in ps
            initplt = scatter!(initplt, [p[1]], [p[2]], [p[3]],
                                    legend=false, color=:black, markersize=5.0,
                                    xlims=fullx, ylims=fully, zlims=fullz)
            frame(anim)
        end
    end
    return gif(anim, "watch$startingtime.gif", fps=framespersecond)
end

function findminima(p0,initialstepsize,tolerance,
                G::ConstraintVariety,
                evaluateobjectivefunctiongradient::Function;
                maxseconds=100, maxlocalsteps=50)
    initialtime = Base.time()
    keepgoing, converged = true, false
    p = copy(p0) # initialize before updating `p` below
    ε0 = initialstepsize # initialize before updating `ε0` below
    ps = [p0] # record the *main steps* from p0, newp, newp, ... until converged
    lastLSR = LocalStepsResult(p,ε0,[],[],[],p,ε0,converged,0,0)

    while keepgoing
        if (Base.time() - initialtime) > maxseconds
            keepgoing = false
            println("We ran out of time... Try setting `maxseconds` to a larger value than $maxseconds")
        end
        # update LSR, only store the *last local run*
        lastLSR = takelocalsteps(p, ε0, tolerance, G, evaluateobjectivefunctiongradient, maxsteps=maxlocalsteps)
        if lastLSR.converged
            keepgoing = false
            converged = true
            push!(ps, lastLSR.newsuggestedstartpoint)
            return OptimizationResult(ps,p0,initialstepsize,tolerance,converged,lastLSR,G,evaluateobjectivefunctiongradient)
        else
            p = lastLSR.newsuggestedstartpoint
            ε0 = lastLSR.newsuggestedstepsize # update and try again!
            push!(ps, p) # record this *main step*
        end
    end
    return OptimizationResult(ps,p0,initialstepsize,tolerance,converged,lastLSR,G,evaluateobjectivefunctiongradient)
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
            plt1 = scatter!(plt1, [q[1]], [q[2]], legend=false, color=:black, xlims=[-2,2], ylims=[-2,2])
        end
        for q in localqs
            plt2 = scatter!(plt2, [q[1]], [q[2]], legend=false, color=:blue, xlims=zoomx, ylims=zoomy)
        end
        vnorms = result.lastlocalstepsresult.allcomputedprojectedgradientvectornorms
        pltvnorms = scatter(vnorms, legend=false, title="norm(v) for last local steps")
        plt = plot(plt1,plt2,pltvnorms, layout=(1,3), size=(900,300) )
        return plt
    elseif dim == 3
        pointz = result.constraintvariety.samples
        mediannorm = Statistics.median([LinearAlgebra.norm(pt) for pt in pointz])
        pointz = filter(x -> LinearAlgebra.norm(x) < 2*mediannorm, pointz)
        plt1,plt2 = plot(), plot() # initialize
        fullx = [minimum([q[1] for q in pointz]) - 0.01, maximum([q[1] for q in pointz]) + 0.01]
        fully = [minimum([q[2] for q in pointz]) - 0.01, maximum([q[2] for q in pointz]) + 0.01]
        fullz = [minimum([q[3] for q in pointz]) - 0.01, maximum([q[3] for q in pointz]) + 0.01]
        globalqs = result.computedpoints
        zoomx = [minimum([q[1] for q in globalqs]) - 0.01, maximum([q[1] for q in globalqs]) + 0.01]
        zoomy = [minimum([q[2] for q in globalqs]) - 0.01, maximum([q[2] for q in globalqs]) + 0.01]
        zoomz = [minimum([q[3] for q in globalqs]) - 0.01, maximum([q[3] for q in globalqs]) + 0.01]
        for p in pointz
            plt1 = scatter!(plt1, [p[1]], [p[2]], [p[3]],
                legend=false, color=:orange, xlims=fullx, ylims=fully, zlims=fullz, markersize=2)
        end
        for q in globalqs
            plt1 = scatter!(plt1, [q[1]], [q[2]], [q[3]],
                legend=false, color=:black, xlims=fullx, ylims=fully, zlims=fullz, markersize=4)
            plt2 = scatter!(plt2, [q[1]], [q[2]], [q[3]],
                legend=false, color=:black, xlims=zoomx, ylims=zoomy, zlims=zoomz)
        end
        vnorms = result.lastlocalstepsresult.allcomputedprojectedgradientvectornorms
        pltvnorms = scatter(vnorms, legend=false, title="norm(v) for last local steps")
        plt = plot(plt1,plt2,pltvnorms, layout=(1,3), size=(900,300) )
        return plt
    end
end;
