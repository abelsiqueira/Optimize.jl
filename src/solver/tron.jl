#  Some parts of this code were adapted from
# https://github.com/PythonOptimizers/NLP.py/blob/develop/nlp/optimize/tron.py

using LinearOperators, NLPModels

export tron

function tron_tuning_problem()
  params = [:line_search_acceptance, :radius_fraction,
            :cauchy_step_factor, :acceptance_threshold,
            :decrease_threshold, :increase_threshold,
            :large_decrease_factor, :small_decrease_factor,
            :increase_factor]
  x0   = [1.0e-2,    1.0, 10.0, 0.0001,   0.25,   0.75,   0.25,    0.5,    4.0]
  lvar = [0.0001, 0.0001,  1.1, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0001]
  uvar = [   0.5,    2.0, 99.9, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 10.0]
  c(x) = [x[1] - x[2]; x[3] - x[4]; x[4] - x[5]; x[7] < x[8]]
  lcon = [-Inf; -Inf; -Inf; -Inf]
  ucon = [0.0; 0.0; 0.0; 0.0]
  return params, x0, lvar, uvar, c, lcon, ucon
end

"""`tron(nlp)`

A pure Julia implementation of a trust-region solver for bound-constrained
optimization:

    min f(x)    s.t.    ℓ ≦ x ≦ u

TRON is described in

Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
"""
function tron(nlp :: AbstractNLPModel;
              line_search_acceptance :: Real = 1e-2,
              radius_fraction        :: Real = 1.0,
              cauchy_step_factor     :: Real = 10.0,
              acceptance_threshold   :: Real = 1e-4,
              decrease_threshold     :: Real = 0.25,
              increase_threshold     :: Real = 0.75,
              large_decrease_factor  :: Real = 0.25,
              small_decrease_factor  :: Real = 0.5,
              increase_factor        :: Real = 4.0,
              verbose :: Bool=false,
              itmax :: Int=10_000 * nlp.meta.nvar,
              max_cgiter :: Int=nlp.meta.nvar,
              cgtol :: Real=0.1,
              max_f :: Int=0,
              max_time :: Real=60.0,
              atol :: Real=1e-8,
              rtol :: Real=1e-6,
              fatol :: Real=0.0,
              frtol :: Real=1e-12
             )
  ℓ = nlp.meta.lvar
  u = nlp.meta.uvar
  f(x) = obj(nlp, x)
  g(x) = grad(nlp, x)
  n = nlp.meta.nvar

  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  iter = 0
  start_time = time()
  el_time = 0.0

  # Preallocation
  temp = zeros(n)
  gpx = zeros(n)
  xc = zeros(n)
  Hs = zeros(n)

  x = max.(ℓ, min.(nlp.meta.x0, u))
  fx = f(x)
  gx = g(x)
  num_success_iters = 0

  # Optimality measure
  project_step!(gpx, x, gx, ℓ, u, -1.0)
  πx = norm(gpx)
  ϵ = atol + rtol * πx
  fmin = min(-1.0, fx) / eps(eltype(x))
  optimal = πx <= ϵ
  tired = iter >= itmax || el_time > max_time || neval_obj(nlp) > max_f
  unbounded = fx < fmin
  stalled = false
  status = :unknown

  αC = 1.0
  tr = TRONTrustRegion(min(max(1.0, 0.1 * norm(πx)), 100.0),
                       acceptance_threshold = acceptance_threshold,
                       decrease_threshold = decrease_threshold,
                       increase_threshold = increase_threshold,
                       small_decrease_factor = small_decrease_factor,
                       large_decrease_factor = large_decrease_factor,
                       increase_factor = increase_factor)
  if verbose
    @printf("%4s  %9s  %7s  %7s  %7s  %s\n", "Iter", "f", "π", "Radius", "Ratio", "CG-status")
    @printf("%4d  %9.2e  %7.1e  %7.1e\n", iter, fx, πx, get_property(tr, :radius))
  end
  while !(optimal || tired || stalled || unbounded)
    # Current iteration
    xc .= x
    fc = fx
    Δ = get_property(tr, :radius)
    H = hess_op!(nlp, xc, temp)

    αC, s, cauchy_status = cauchy(x, H, gx, Δ, αC, ℓ, u,
                   line_search_acceptance=line_search_acceptance,
                   radius_fraction=radius_fraction,
                   step_factor=cauchy_step_factor)

    if cauchy_status != :success
      status = cauchy_status
      stalled = true
      continue
    end

    s, Hs, cgits, cginfo = projected_newton!(x, H, gx, Δ, cgtol, s, ℓ, u, max_cgiter=max_cgiter)
    slope = dot(gx, s)
    qs = 0.5 * dot(s, Hs) + slope
    fx = f(x)

    try
      ratio!(tr, nlp, fc, fx, qs, x, s, slope)
    catch exc
      status = :neg_pred
      stalled = true
      continue
    end

    s_norm = norm(s)
    if num_success_iters == 0
      tr.radius = min(Δ, s_norm)
    end

    # Update the trust region
    update!(tr, s_norm)

    if acceptable(tr)
      num_success_iters += 1
      gx = g(x)
      project_step!(gpx, x, gx, ℓ, u, -1.0)
      πx = norm(gpx)
    end

    # No post-iteration

    if !acceptable(tr)
      fx = fc
      x .= xc
    end

    iter += 1
    el_time = time() - start_time
    tired = iter >= itmax || el_time > max_time || neval_obj(nlp) > max_f
    optimal = πx <= ϵ
    unbounded = fx < fmin

    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  %7.1e  %s\n", iter, fx, πx, Δ, Ared / Pred, cginfo)
  end

  if tired
    status = iter >= itmax ? :max_iter : :max_time
  elseif optimal
    status = :first_order
  elseif unbounded
    status = :unbounded
  end

  return ExecutionStats(status, x=x, f=fx, normg=πx, iter=iter, time=el_time,
                        eval=deepcopy(counters(nlp)))
end

"""`s = projected_line_search!(x, H, g, d, ℓ, u; line_search_acceptance = 1e-2)`

Performs a projected line search, searching for a step size `t` such that

    0.5sᵀHs + sᵀg ≦ line_search_acceptance * sᵀg,

where `s = P(x + t * d) - x`, while remaining on the same face as `x + d`.
Backtracking is performed from t = 1.0. `x` is updated in place.
"""
function projected_line_search!{T <: Real}(x::AbstractVector{T},
                                           H::Union{AbstractMatrix,AbstractLinearOperator},
                                           g::AbstractVector{T},
                                           d::AbstractVector{T},
                                           ℓ::AbstractVector{T},
                                           u::AbstractVector{T};
                                           line_search_acceptance::Real = 1e-2)
  α = one(T)
  _, brkmin, _ = breakpoints(x, d, ℓ, u)
  nsteps = 0
  n = length(x)

  s = zeros(n)
  Hs = zeros(n)

  search = true
  while search && α > brkmin
    nsteps += 1
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
    if qs <= line_search_acceptance * slope
      search = false
    else
      α /= 2
    end
  end
  if α < 1.0 && α < brkmin
    α = brkmin
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
  end

  project_step!(s, x, d, ℓ, u, α)
  x .= x .+ s

  return s
end

"""`α, s = cauchy(x, H, g, Δ, ℓ, u; line_search_acceptance = 1e-2,
                  radius_fraction = 1.0, step_factor=10.0)`

Computes a Cauchy step `s = P(x - α g) - x` for

    min  q(s) = ¹/₂sᵀHs + gᵀs     s.t.    ‖s‖ ≦ radius_fraction * Δ,  ℓ ≦ x + s ≦ u,

with the sufficient decrease condition

    q(s) ≦ line_search_acceptance * sᵀg.
"""
function cauchy{T <: Real}(x::AbstractVector{T},
                           H::Union{AbstractMatrix,AbstractLinearOperator},
                           g::AbstractVector{T},
                           Δ::Real, α::Real, ℓ::AbstractVector{T}, u::AbstractVector{T};
                           line_search_acceptance::Real = 1e-2,
                           radius_fraction::Real = 1.0,
                           step_factor::Real = 10.0)
  # TODO: Use brkmin to care for g direction
  _, _, brkmax = breakpoints(x, -g, ℓ, u)
  n = length(x)
  s = zeros(n)
  Hs = zeros(n)

  project_step!(s, x, g, ℓ, u, -α)

  # Interpolate or extrapolate
  s_norm = norm(s)
  if s_norm > radius_fraction * Δ
    interp = true
  else
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
    interp = qs >= line_search_acceptance * slope
  end

  status = :success

  if interp
    search = true
    while search
      α /= step_factor
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = norm(s)
      if s_norm <= radius_fraction * Δ
        slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
        search = qs >= line_search_acceptance * slope
      end
      # TODO: Correctly assess why this fails
      if α < sqrt(nextfloat(zero(α)))
        stalled = true
        status = :smallstep
        search = false
      end
    end
  else
    search = true
    αs = α
    while search && α <= brkmax
      α *= step_factor
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = norm(s)
      if s_norm <= radius_fraction * Δ
        slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
        if qs <= line_search_acceptance * slope
          αs = α
        end
      else
        search = false
      end
    end
    # Recover the last successful step
    α = αs
    s = project_step!(s, x, g, ℓ, u, -α)
  end
  return α, s, status
end

"""`projected_newton!(x, H, g, Δ, gctol, s, max_cgiter, ℓ, u)`

Compute an approximate solution `d` for

min q(d) = ¹/₂dᵀHs + dᵀg    s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ

starting from `s`.  The steps are computed using the conjugate gradient method
projected on the active bounds.
"""
function projected_newton!{T <: Real}(x::AbstractVector{T}, H::Union{AbstractMatrix,AbstractLinearOperator},
                                      g::AbstractVector{T}, Δ::Real, cgtol::Real, s::AbstractVector{T},
                                      ℓ::AbstractVector{T}, u::AbstractVector{T};
                                      max_cgiter::Int = max(50, length(x)))
  n = length(x)
  status = ""

  Hs = H * s

  # Projected Newton Step
  exit_optimal = false
  exit_pcg = false
  exit_itmax = false
  iters = 0
  x .= x .+ s
  project!(x, x, ℓ, u)
  while !(exit_optimal || exit_pcg || exit_itmax)
    ifree = setdiff(1:n, active(x, ℓ, u))
    if length(ifree) == 0
      exit_optimal = true
      continue
    end
    Z = opExtension(ifree, n)
    @views wa = g[ifree]
    @views gfree = Hs[ifree] + wa
    gfnorm = norm(wa)

    ZHZ = Z' * H * Z
    st, stats = Krylov.cg(ZHZ, -gfree, radius=Δ, rtol=cgtol, atol=0.0,
                          itmax=max_cgiter)
    iters += length(stats.residuals)
    status = stats.status

    # Projected line search
    @views xfree = x[ifree]
    @views w = projected_line_search!(xfree, ZHZ, gfree, st, ℓ[ifree], u[ifree])
    @views s[ifree] += w

    Hs .= H * s

    @views gfree .= Hs[ifree] .+ g[ifree]
    if norm(gfree) <= cgtol * gfnorm
      exit_optimal = true
    elseif status == "on trust-region boundary"
      exit_pcg = true
    elseif iters >= max_cgiter
      exit_itmax = true
    end
  end
  status = if exit_optimal
    "stationary point found"
  elseif exit_itmax
    "maximum number of iterations"
  else
    status # on trust-region
  end

  return s, Hs, iters, status
end
