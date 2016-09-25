using Optimize
using FactCheck
using NLPModels
using AmplNLReader
using OptimizationProblems
using Compat
import Compat.String

# Test TRON
include("solvers/tron.jl")

models = [AmplModel("dixmaanj.nl"), MathProgNLPModel(dixmaanj(), name="dixmaanj")]

@static if is_unix()
  using CUTEst
  push!(models, CUTEstModel("DIXMAANJ", "-param", "M=30"))
end
solvers = [:trunk, :lbfgs, :tron]

for model in models
  for solver in solvers
    stats = run_solver(solver, model, verbose=false)
    assert(all([stats...] .>= 0))
    reset!(model)
  end
end

@static if is_unix()
  for m in models
    if typeof(m) == CUTEst.CUTEstModel
      cutest_finalize(m)
    end
  end
end

# clean up the test directory
@static if is_unix()
  here = dirname(@__FILE__)
  so_files = filter(x -> (ismatch(r".so$", x) || ismatch(r".dylib$", x)), readdir(here))

  for so_file in so_files
    rm(joinpath(here, so_file))
  end

  rm(joinpath(here, "AUTOMAT.d"))
  rm(joinpath(here, "OUTSDIF.d"))
end

# test benchmark helpers, skip constrained problems (hs7 has constraints)
run_ampl_problem(:trunk, :dixmaanj, 0, verbose=true, monotone=false)
probs = [:dixmaane, :dixmaanf, :dixmaang, :dixmaanh, :dixmaani, :dixmaanj, :hs7]
bmark_and_profile(solvers, probs, 99, bmark_args=Dict{Symbol, Any}(:skipif => m -> m.meta.ncon > 0))
