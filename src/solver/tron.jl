using LinearOperators, NLPModels
# A trust-region solver for unconstrained optimization with bounds

export tron

function active(x, l, u; ϵlu::Real = 1e-6)
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

function tron(nlp :: AbstractNLPModel; μ₀ :: Real=1e-2,
    μ₁ :: Real=1.0, σ :: Real=1.2, verbose=false, itmax :: Integer=100000,
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

  # Projection
  P!(y, x, a, v) = begin # y = P[x + a*v] - x, where l ≦ x ≦ u
    for i = 1:n
      y[i] = a*v[i] > 0 ? min(x[i]+a*v[i], u[i])-x[i] : max(x[i]+a*v[i], l[i])-x[i]
    end
    return y
  end

  # Preallocation
  xcur = zeros(n)
  dcur = zeros(n)
  sα = zeros(n)
  sαn = zeros(n)
  sαp = zeros(n)
  wβ = zeros(n)
  gpx = zeros(n)

  x = max(min(nlp.meta.x0, u), l)
  gx = g(x)
  qdcur = Inf

  # Optimality measure
  P!(gpx, x, -1.0, gx)
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
    q(d) = 0.5*dot(d, H * d) + dot(d, gx)
    # Projected step
    P!(sα, x, -α, gx)
    copy!(sαp, sα)

    # TODO: Compute the breakpoints and use them to prevent computing hprod too
    # often
    # Find α satisfying the decrease condition increasing if it's
    # possible, or decreasing if necessary.
    if q(sα) <= μ₀*dot(gx,sα) && norm(sα) <= μ₁*Δ
      while q(sα) <= μ₀*dot(gx,sα) && norm(sα) <= μ₁*Δ
        α *= σ
        P!(sαn, x, -α, gx)
        # Check if the step is in a corner.
        norm(sαn - sα) < 1e-12 && break
        copy!(sαp, sα)
        copy!(sα, sαn)
        if α > 1e12
          stalled = true
          status = "α too large"
          break
        end
      end
      copy!(sα, sαp)
      α /= σ
    else
      while q(sα) > μ₀*dot(gx,sα) || norm(sα) > μ₁*Δ
        α /= σ
        copy!(sαp, sα)
        P!(sα, x, -α, gx)
        if α < 1e-24
          stalled = true
          status = "α too small"
          break
        end
      end
    end

    stalled && break

    Δcur = norm(sα)
    copy!(dcur, sα)
    copy!(xcur, x)
    BLAS.axpy!(1.0, sα, xcur)

    qdcur = q(dcur)

    nsmall = 0

    cgtol = max(ϵ, min(0.7 * cgtol, 0.01 * π))
    while Δcur < Δ
      A = active(xcur, l, u)
      if length(A) == nlp.meta.nvar
        break
      end
      I = setdiff(1:n, A)
      Z = ExtensionOperator(I, n)
      v = H * dcur + gx
      if norm(Z'*v) < ϵ
        break
      end
      st, stats = Krylov.cg(Z'*H*Z, -(Z'*v), radius=Δ-Δcur, atol=cgtol, rtol=0.0, itmax=max(2*n, 50))
      st = Z*st
      β = 1.0
      P!(wβ, xcur, β, st)
      if norm(wβ) < ϵ
        break
      end
      while q(dcur + wβ) > qdcur + μ₀*min(dot(v, wβ), 0)
        β *= 0.9
        P!(wβ, xcur, β, st)
      end
      Δcur += norm(wβ)
      BLAS.axpy!(1.0, wβ, dcur)
      qdcur = q(dcur)
      BLAS.axpy!(1.0, wβ, xcur)
      if norm(wβ) < 1e-3*Δ
        nsmall += 1
        if nsmall == 3
          break
        end
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
      P!(gpx, x, -1.0, gx)
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

