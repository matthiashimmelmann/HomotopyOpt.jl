# Tests for HomotopyOpt.jl

To run tests, navigate to the projects folder and activate the project's local environment. Afterwards, simply run the command `test`.

```
julia> cd("<your_julia_home_folder>\\HomotopyOpt.jl")
julia> pwd()      # Print the cursor's current location
"<your_julia_home_folder>\\HomotopyOpt.jl"
julia> ]          # Pressing ] let's us enter Julia's package manager
(@v1.6) pkg> activate .
(HomotopyOpt) pkg> test
```

At the moment, this runs tests in 2D and in 3D for each of the optimization methods `gaussnewtonstep`, `EDStep` and `twostep`.