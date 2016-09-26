# Some parts of this code were adapted from
# https://github.com/PythonOptimizers/NLP.py/blob/develop/nlp/optimize/tron.py

using LinearOperators, NLPModels

export tron

function active(x, l, u; ϵlu::Real = 1e-8)
  A = Int[]
  n = length(x)
  for i = 1:n
    if x[i] < l[i] + ϵlu * (u[i]-l[i])
      push!(A, i)
    elseif x[i] > u[i] - ϵlu * (u[i]-l[i])
      push!(A, i)
    end
  end
  return A
end

"""
    tron(nlp)

A trust-region solver for bound-constrained optimization.

A pure Julia implementation of TRON as described in

Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
"""
function tron(nlp :: AbstractNLPModel; μ₀ :: Real=1e-2,
    μ₁ :: Real=1.0, σ :: Real=10, verbose=false, itmax :: Integer=100000,
    timemax :: Real=60, mem :: Integer=5, atol :: Real=1e-8, rtol :: Real=1e-6,
    σ₁ = 0.25, σ₂ = 0.5, σ₃ = 4.0, η₀ = 1e-3, η₁ = 0.25, η₂ = 0.75)
  l = nlp.meta.lvar
  u = nlp.meta.uvar
  f(x) = obj(nlp, x)
  g(x) = grad(nlp, x)
  n = nlp.meta.nvar

  status = ""
  iter = 0
  start_time = time()
  el_time = 0.0

  # Preallocation
  xcur = zeros(n)
  dcur = zeros(n)
  s = zeros(n)
  sn = zeros(n)
  sp = zeros(n)
  gpx = zeros(n)

  x = project(nlp.meta.x0, l, u)
  gx = g(x)
  qs = Inf
  qdcur = Inf

  # Optimality measure
  project_step!(gpx, x, -gx, l, u)
  πx = norm(gpx)
  ϵ = atol + rtol * π
  cgtol = 1.0
  optimal = πx <= ϵ
  tired = iter >= itmax ||  el_time > timemax
  stalled = false

  α = 1.0
  fx = f(x)
  Δ = max(0.1, 0.1*norm(gx))
  ρ = Inf
  if verbose
    @printf("%4s  %9s  %7s  %7s  %8s\n", "Iter", "f", "π", "Radius", "Ratio")
    @printf("%4d  %9.2e  %7.1e  %7.1e\n", iter, fx, πx, Δ)
  end
  while !(optimal || tired || stalled)
    # Model
    H = hess_op(nlp, x)
    q(d, Hd, slope) = 0.5*dot(d, Hd) + slope
    # Projected step
    project_step!(s, x, -α*gx, l, u)
    copy!(sp, s)

    _, _, brkmax = breakpoints(x, -gx, l, u)
    # Find α satisfying the decrease condition increasing if it's
    # possible, or decreasing if necessary.
    s_norm = norm(s)
    if s_norm > μ₁*Δ
      interp = true
    else
      slope = dot(s, gx)
      Hs = H * s
      qs = q(s, Hs, slope)
      interp = qs >= μ₀ * slope
    end

    if interp
      search = true
      while search
        α /= σ
        project_step!(s, x, -α*gx, l, u)
        s_norm = norm(s)
        if s_norm <= μ₁*Δ
          Hs = H * s
          slope = dot(s, gx)
          qs = q(s, Hs, slope)
          search = qs > μ₀ * slope
        end
        if α < 1e-24
          stalled = true
          status = "α too small"
          break
        end
      end
    else
      search = true
      copy!(sp, s)
      while search && α < brkmax
        α *= σ
        project_step!(s, x, -α*gx, l, u)
        # Check if the step is in a corner.
        s_norm = norm(s)
        if s_norm <= μ₁*Δ
          Hs = H * s
          slope = dot(s, gx)
          qs = q(s, Hs, slope)
          if qs < μ₀ * slope
            copy!(sp, s)
          else
            search = false
          end
        else
          search = false
        end
        if α > 1e12
          stalled = true
          status = "α too large"
          break
        end
      end
      copy!(s, sp)
      Hs = H * s
      slope = dot(s, gx)
      qs = q(s, Hs, slope)
    end

    stalled && break

    Δcur = norm(s)
    copy!(dcur, s)
    copy!(xcur, x)
    BLAS.axpy!(1.0, s, xcur)
    project!(xcur, x + s, l, u)

    qdcur = qs
    nsmall = 0

    # Projected Newton Step
    exit_optimal = false
    exit_pcg = Δcur >= Δ
    exit_itmax = false
    cgtol = max(ϵ, min(0.7 * cgtol, 0.01 * π))
    newton_itmax = n
    newton_iter = 0
    while !(exit_optimal || exit_pcg || exit_itmax)
      A = active(xcur, l, u)
      if length(A) == nlp.meta.nvar
        exit_optimal = true
        continue
      end
      Z = ExtensionOperator(setdiff(1:n, A), n)
      v = H * dcur + gx
      gfree = Z'*gx
      vfree = Z'*v
      if norm(vfree) < ϵ * norm(gfree)
        exit_optimal = true
        continue
      end
      ZHZ = Z' * H * Z
      st, stats = Krylov.cg(ZHZ, -vfree, radius=Δ-Δcur, atol=cgtol, rtol=0.0, itmax=max(2*n, 50))
      # TODO: When Krylov.stats get iter, sum number of cg iters.
      newton_iter += 1
      # Projected line search
      st = Z*st
      β = 1.0
      project_step!(s, xcur, β*st, l, u)
      if norm(s) < ϵ
        exit_pcg = true
        continue
      end
      Hs = H * s
      slope = dot(v, s)
      qs = q(s, Hs, slope)
      _, brkmin, _ = breakpoints(xcur, st, l, u)

      search = true
      while search && β > brkmin
        if qs <= μ₀*dot(v, s)
          search = false
          continue
        end
        β *= 0.9
        project_step!(s, xcur, β*st, l, u)
        Hs = H * s
        slope = dot(v, s)
        qs = q(s, Hs, slope)
      end
      if β < 1.0 && β < brkmin
        β = brkmin
        project_step!(s, xcur, β*st, l, u)
        Hs = H * s
        slope = dot(v, s)
        qs = q(s, Hs, slope)
      end

      BLAS.axpy!(1.0, s, dcur)
      BLAS.axpy!(1.0, s, xcur)

      qdcur += qs

      Δcur = norm(dcur)

      v = H * dcur + gx
      if norm(Z'*v) <= cgtol * norm(Z'*gx)
        exit_optimal = true
      elseif stats.status == "on trust-region boundary"
        exit_pcg = true
      elseif newton_iter >= newton_itmax
        exit_itmax = true
      end

    end

    # Candidate
    fxcur = f(xcur)
    slope = dot(dcur, gx)

    # Ratio
    try
      ρ = ratio(nlp, fx, fxcur, qdcur, xcur, dcur, slope)
    catch e
      # Failed
      status = e.msg
      stalled = true
      break
    end

    fxold = fx

    # Update x
    if ρ >= η₀
      copy!(x, xcur)
      fx = fxcur

      y = gx
      gx = g(x)
      project_step!(gpx, x, -gx, l, u)
      πx = norm(gpx)
    end

    # Update the trust region
    θ = dot(gx,dcur)
    nrmd = norm(dcur)
    γ = fx - fxold - θ
    if γ <= eps(θ)
      αstar = Inf
    else
      αstar = -nrmd*θ/2γ
    end

    if ρ <= η₁
      Δ = max(min(αstar, σ₂*Δ), σ₁*min(nrmd,Δ))
    elseif ρ < η₂
      Δ = max(min(αstar, σ₃*Δ), σ₁*Δ)
    else
      Δ = max(min(αstar, σ₃*Δ), Δ)
    end

    iter += 1
    el_time = time() - start_time
    tired = iter >= itmax ||  el_time > timemax
    optimal = πx <= ϵ

    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  %8.1e\n", iter, fx, πx, Δ, ρ)
  end

  if tired
    status = iter >= itmax ? "maximum number of iterations" : "maximum elapsed time"
  elseif !stalled
    status = "first-order stationary"
  elseif stalled && status == ""
    status = "stalled"
  end

  return x, f(x), πx, iter, optimal, tired, status, el_time
end

"""
    nbrk, brkmin, brkmax = breakpoints(x, d, l, u)

Find the smallest and largest values of α such that x + αd lies on the boundary.
We assume x is feasible.
"""
function breakpoints(x, d, l, u)
  n = length(x)
  pos = find( (d .> 0) & (x .< u) )
  neg = find( (d .< 0) & (x .> l) )

  nbrk = length(pos) + length(neg)
  nbrk == 0 && return 0, 0, 0

  brkmin = Inf
  brkmax = 0.0
  if length(pos) > 0
    steps = (u[pos] - x[pos])./d[pos]
    @assert all(steps .> 0)
    brkmin = min(brkmin, minimum(steps))
    brkmax = max(brkmax, maximum(steps))
  end
  if length(neg) > 0
    steps = (l[neg] - x[neg])./d[neg]
    @assert all(steps .> 0)
    brkmin = min(brkmin, minimum(steps))
    brkmax = max(brkmax, maximum(steps))
  end
  return nbrk, brkmin, brkmax
end

function project(x, l, u)
  y = zeros(x)
  return project!(y, x, l, u)
end

function project!(y, x, l, u)
  n = length(x)
  for i = 1:n
    y[i] = max(l[i], min(x[i], u[i]))
  end
  return y
end

function project_step(x, d, l, u)
  y = zeros(x)
  return project_step!(y, x, d, l, u)
end

function project_step!(y, x, d, l, u)
  n = length(x)
  for i = 1:n
    y[i] = max(l[i], min(x[i] + d[i], u[i])) - x[i]
  end
  return y
end
