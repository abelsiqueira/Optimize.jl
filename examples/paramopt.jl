using CUTEst, Optimize, BenchmarkProfiles, Plots
pyplot()

#= This is an example of parameter optimization on the solver trunk.
=#
function paramopt(solver = trunk)
  ps = CUTEst.select(max_var=10, max_con=0, only_free_var=true)
  problems = (CUTEstModel(p) for p in ps)

  optimal_parameters = tune(solver, problems,
                            bbtol=1e-4, bbmax_f=50, verbose=true,
                            solver_args = Dict(:max_f=>1000, :max_time=>3.0))

  optsolver = solver_with_parameters(solver, optimal_parameters)

  bmark_and_profile([solver, optsolver], problems,
                    bmark_args=Dict(:verbose=>false, :max_f=>1000, :max_time=>3.0))
  png("$solver-vs-opt$solver")
end

paramopt()
