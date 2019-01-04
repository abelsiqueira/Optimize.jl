include("blackbox.jl")

export tune, solver_with_parameters

"""`optimal_parameters = tune(solver, problems)`

Optimize parameters of `solver` for `problems`. Currently restricted to
- The parameters, bounds and constraints described by the solver itself;
- The blackbox evaluation is the sum of counters of the problem weighted
by 10 if the solver declared failure for that problem.

Disclaimer: For use with CUTEst, do
```
ps = CUTEst.select(...)
problems = (CUTEstModel(p) for p in ps)
```
"""
function tune(solver :: Function, problems :: Any;
              tuning_problem_function = eval(parse("$(solver)_tuning_problem")),
              parameters :: Array{Symbol} = [:all],
              initial :: Array{<: Any} = [],
              lower :: Vector = Float64[],
              upper :: Vector = Float64[],
              constraints :: Union{Function,Nothing} = nothing,
              conslower :: Vector = Float64[],
              consupper :: Vector = Float64[],
              flags :: Array{Symbol} = [:auto],
              bbtol :: Real = 1e-2,
              bbmax_f :: Real = 100,
              verbose :: Bool=true,
              bbverbose :: Bool=false,
              solver_args :: Dict{Symbol,<: Any} = Dict()
             )

  s = Symbol( split(string(solver),".")[end] )
  params, x0, lvar, uvar, c, lcon, ucon = tuning_problem_function()

  if parameters != [:all] || flags != [:auto] || length(conslower) > 0 || length(consupper) > 0
    if length(initial) > 0 && length(initial) != length(parameters)
      error("Initial values are given but don't match the parameters")
    elseif !issubset(parameters, params)
      s = join(map(x->":$x", setdiff(parameters, params)), ", ")
      error("Unrecognized parameters '$s'")
    end

    # :auto will use replace by default
    if :auto in flags
      length(flags) > 1 && error(":auto in flags is supposed to be used alone")
      if length(conslower) > 0 || length(consupper) > 0
        push!(flags, :replace_constraints)
      end
      if length(parameters) > 0 || length(initial) > 0 || length(lower) > 0 || length(upper) > 0
        push!(flags, :replace_variables)
      end
    end

    if :update in flags
      push!(flags, :update_variables)
      push!(flags, :update_constraints)
    end
    if :replace in flags
      push!(flags, :replace_variables)
      push!(flags, :replace_constraints)
    end
    if :replace_variables in flags && :update_variables in flags
      error("Conflicting flags: replacing and updating variables")
    elseif :replace_constraints in flags && :update_constraints in flags
      error("Conflicting flags: replacing and updating constraints")
    end

    J = [findfirst(params .== p) for p in parameters]
    if :update_variables in flags
      for i = 1:length(initial)
        x0[J[i]] = initial[i]
      end
      for i = 1:length(lower)
        lvar[J[i]] = lower[i]
      end
      for i = 1:length(upper)
        uvar[J[i]] = upper[i]
      end
    elseif :replace_variables in flags
      # Whatever is not given, is constructed by reducing
      x0 = length(initial) > 0 ? initial : x0[J]
      lvar = length(lower) > 0 ? lower : lvar[J]
      uvar = length(upper) > 0 ? upper : uvar[J]
      params = parameters
      # TODO: Allow replacing variables without losing constraints
      if !(:replace_constraints in flags)
        lcon, ucon, c = zeros(0), zeros(0), ()->()
      end
    end
    if :update_constraints in flags
      for i = 1:length(conslower)
        lcon[i] = conslower[i]
      end
      for i = 1:length(consupper)
        ucon[i] = consupper[i]
      end
      if constraints != nothing
        error("To change the constraints, use :replace_constraints")
      end
    elseif :replace_constraints in flags
      params, lcon, ucon = parameters, conslower, consupper
      c = constraints == nothing ? ()->() : get(constraints)
    end
    # TODO: Allow adding additional constraints
  end

  NP = length(params)

  if verbose
    println("Parameter optimization of $solver")
    println("  Parameters under consideration:")
    for i = 1:NP
      @printf("  %+14.8e  ≦  %10s  ≦  %+14.8e\n",
              lvar[i], params[i], uvar[i])
    end
    if length(lcon) > 0
      println("  There are also constraints, which can't be described here")
    else
      println("  No further constraints")
    end
    println("Iteration print")
    println("")
    @printf("%6s  (%9s", "BBEval", "param1")
    for i = 2:NP
      @printf("  %9s", "param$i")
    end
    @printf(")  %15s\n", "f(x)")
  end
  bbiter = 1
  f(x) = begin
    s = 0.0
    try
      stats = solve_problems(solver, problems; verbose=false,
                             solver_args...,
                             (params[i]=>x[i] for i = 1:NP)...)
      for stat in stats
        # TODO: Allow different cost functions
        s += sum_counters(stat.eval) * (stat.status == :first_order ? 1 : 10)
      end
    catch ex
      println("Unhandled exception on tuning: $ex")
      s = Inf
    end

    if verbose
      @printf("%6d  (%+8.2e", bbiter, x[1])
      bbiter += 1
      for i = 2:NP
        @printf("  %+8.2e", x[i])
      end
      @printf(")  %+14.8e\n", s)
    end
    return s
  end
  tnlp = ADNLPModel(f, x0, lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)
  x, _ = blackbox(tnlp, verbose=bbverbose, max_f=bbmax_f, tol=bbtol)

  if verbose
    println("Optimal parameter choice:")
    for i = 1:NP
      println("  $(params[i]) = $(x[i])")
    end
  end

  return (params[i]=>x[i] for i = 1:NP) # optimal_parameters
end

tune(s :: Symbol, p :: Any; kwargs...) = tune(eval(s), p; kwargs...)

"""`altsolver = solver_with_parameters(solver, parameters)`

This function returns `solver` with different parameters, instead of the
default ones. The returned `altsolver` receives the same positional and keyword
arguments as the original `solver`. A possible value for `parameters` would
be the output of `tune`.
"""
function solver_with_parameters(solver :: Function, optimal_parameters)
  return (nlp::AbstractNLPModel, args...; kwargs...) ->
          solver(nlp, args...; optimal_parameters..., kwargs...)
end
