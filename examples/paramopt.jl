using CUTEst, Optimize, BenchmarkProfiles, Plots
pyplot()

#= This is an example of parameter optimization on the solver trunk.
=#
function paramopt()
  ps = CUTEst.select(max_var=10, max_con=0, only_free_var=true)
  problems = (CUTEstModel(p) for p in ps)

  optimal_parameters = tune(trunk, problems,
                            bbtol=1e-4, bbmax_f=50, verbose=true,
                            solver_args = Dict(:max_f=>1000, :max_time=>3.0))

  opttrunk = solver_with_parameters(trunk, optimal_parameters)

  bmark_and_profile([trunk, opttrunk], problems,
                    bmark_args=Dict(:verbose=>false, :max_f=>1000, :max_time=>3.0))
  png("trunk-vs-opttrunk")
end

paramopt()
