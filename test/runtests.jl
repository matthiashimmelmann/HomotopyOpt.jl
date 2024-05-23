using HomotopyOpt, Test

@testset "HomotopyOpt" begin
    include("2DTests.jl")
    include("3DTests.jl")
    include("testEDRetraction.jl")
end
