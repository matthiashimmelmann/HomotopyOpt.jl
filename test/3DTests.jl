using HomotopyContinuation

@testset "whitney umbrella" begin
    @var x y z
    V = HomotopyOpt.ConstraintVariety([x,y,z], [x^2-y^2*z], 3, 2, 100);
    p0 = V.samples[1]
    objective(x) = sin(x[1])+cos(x[2])+cos(sin(x[3]))

    println("gaussnewtonstep")
    @time resultminimum = HomotopyOpt.findminima(p0, 1e-4, V, objective; maxseconds=120, whichstep="gaussnewtonstep", initialstepsize=0.5);
    @test(resultminimum.converged==true)
    @test(resultminimum.lastpointisminimum==true)

    println("EDStep")
    @time resultminimum = HomotopyOpt.findminima(p0, 1e-4, V, objective; maxseconds=120, whichstep="EDStep", initialstepsize=0.5);
    @test(resultminimum.converged==true)
    @test(resultminimum.lastpointisminimum==true)

    println("twostep")
    @time resultminimum = HomotopyOpt.findminima(p0, 1e-4, V, objective; maxseconds=120, whichstep="twostep", initialstepsize=0.5);
    @test(resultminimum.converged==true)
    @test(resultminimum.lastpointisminimum==true)
end
