export lbfgs

function lbfgs_tuning_problem()
  params = [:linesearch_acceptance, :linesearch_angle]
  x0     = [1.0e-4, 0.9999]
  lvar   = [1.0e-4, 0.9]
  uvar   = [   0.9, 0.9999]
  c(x) = Float64[]
  lcon = Float64[]
  ucon = Float64[]
  return params, x0, lvar, uvar, c, lcon, ucon
end

function lbfgs(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_f :: Int=0,
               max_time :: Float64=Inf,
               verbose :: Bool=true,
               linesearch_acceptance :: Real = 1.0e-4,
               linesearch_angle      :: Real = 0.9999,
               mem :: Int=5)

  start_time = time()
  elapsed_time = 0.0

  x = copy(nlp.meta.x0)
  n = nlp.meta.nvar

  xt = Array{Float64}(n)
  ∇ft = Array{Float64}(n)

  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  H = InverseLBFGSOperator(n, mem, scaling=true)

  ∇fNorm = BLAS.nrm2(n, ∇f, 1)
  ϵ = atol + rtol * ∇fNorm
  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  iter = 0

  verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
  verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

  optimal = ∇fNorm <= ϵ
  tired = neval_obj(nlp) > max_f || elapsed_time > max_time
  stalled = false

  h = LineModel(nlp, x, ∇f)

  while !(optimal || tired || stalled)
    d = - H * ∇f
    slope = BLAS.dot(n, d, 1, ∇f, 1)
    if slope >= 0.0
      status = :not_descent
      stalled = true
      continue
    end

    verbose && @printf("  %8.1e", slope)

    redirect!(h, x, d)
    # Perform improved Armijo linesearch.
    t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft,
                                              τ₀=linesearch_acceptance,
                                              τ₁=linesearch_angle,
                                              bk_max=25, verbose=false)

    verbose && @printf("  %4d\n", nbk)

    BLAS.blascopy!(n, x, 1, xt, 1)
    BLAS.axpy!(n, t, d, 1, xt, 1)
    good_grad || (∇ft = grad!(nlp, xt, ∇ft))

    # Update L-BFGS approximation.
    push!(H, t * d, ∇ft - ∇f)

    # Move on.
    x = xt
    f = ft
    BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
    # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    iter = iter + 1

    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal = ∇fNorm <= ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_f || elapsed_time > max_time
  end
  verbose && @printf("\n")

  if optimal
    status = :first_order
  elseif tired
    if neval_obj(nlp) > max_f
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return ExecutionStats(status, x=x, f=f, normg=∇fNorm, iter=iter, time=elapsed_time,
                        eval=deepcopy(counters(nlp)))
end
