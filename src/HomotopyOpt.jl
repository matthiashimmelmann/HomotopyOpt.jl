module HomotopyOpt

import HomotopyContinuation
import LinearAlgebra
import ImplicitPlots: implicit_plot
import Plots
import Statistics
import Implicit3DPlotting: plot_implicit_surface, plot_implicit_surface!, plot_implicit_curve, plot_implicit_curve!, GLMakiePlottingLibrary

export ConstraintVariety,
       findminima,
       watch,
       draw

struct ConstraintVariety
    variables
    equations
    jacobian
    ambientdimension
    dimensionofvariety
    samples
    implicitequations

    function ConstraintVariety(eqnz, N::Int, d::Int, numsamples::Int)
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
        new(varz,algeqnz,dg,N,d,Ωs,eqnz)
    end

    function ConstraintVariety(eqnz,N::Int,d::Int)
        HomotopyContinuation.@var varz[1:N]
        algeqnz = [eqn(varz) for eqn in eqnz]
        dg = HomotopyContinuation.differentiate(algeqnz, varz)
        new(varz,algeqnz,dg,N,d,[],eqnz)
    end
end

function computesystem(p, G::ConstraintVariety,
                evaluateobjectivefunctiongradient::Function)

    dgp = HomotopyContinuation.ModelKit.evaluate(G.jacobian, G.variables => p)
    Up,_ = LinearAlgebra.qr( LinearAlgebra.transpose(dgp) )
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
            push!(ns, LinearAlgebra.norm(vq))
            if LinearAlgebra.norm(vq) < tolerance
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
        pltvnorms = Plots.scatter(vnorms, legend=false, title="norm(v) for last local steps")
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
        GLMakiePlottingLibrary.scatter!(ax3,vnorms; legend=false)

        return fig
    end
end

end
