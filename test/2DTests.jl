using HomotopyContinuation

@testset "circle Test" begin
    @var x y
    V = HomotopyOpt.ConstraintVariety([x,y], [(x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y], 2, 1, 100);
    p0 = V.samples[1]
    objective(x) = sin(x[1])+cos(x[2])+1

    println("gaussnewtonstep")
    @time resultminimum = HomotopyOpt.findminima(p0, 1e-4, V, objective; maxseconds=100, whichstep="gaussnewtonstep", initialstepsize=0.5);
    @test(resultminimum.converged==true)
    @test(resultminimum.lastpointisminimum==true)

    println("EDStep")
    @time resultminimum = HomotopyOpt.findminima(p0, 1e-4, V, objective; maxseconds=100, whichstep="EDStep", initialstepsize=0.5);
    @test(resultminimum.converged==true)
    @test(resultminimum.lastpointisminimum==true)

    println("twostep")
    @time resultminimum = HomotopyOpt.findminima(p0, 1e-4, V, objective; maxseconds=100, whichstep="twostep", initialstepsize=0.5);
    @test(resultminimum.converged==true)
    @test(resultminimum.lastpointisminimum==true)
end
