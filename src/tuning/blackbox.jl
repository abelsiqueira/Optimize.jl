export blackbox

include("mads.jl")

"""`blackbox(nlp)`

Implementation of a blackbox method used for parameter optimization.

At the moment, OrthoMADS is implemented, following the description of

    Mark A. Abramson, Charles Audet, J. E. Dennis Jr., and Sébastien Le Digabel.
    OrthoMADS: A Deterministic MADS Instance with Orthogonal Directions.
    SIAM Journal on Optimization, 2009, Vol. 20, No. 2, pp. 948-966.
"""
function blackbox(nlp :: AbstractNLPModel;
                  tol :: Float64=1e-4,
                  verbose :: Bool=true,
                  max_f :: Int=5000
                 )

  f(x) = obj(nlp, x)
  feasible(x) = if unconstrained(nlp)
    true
  else
    bl, bu, cl, cu = [getfield(nlp.meta, f) for f in [:lvar,:uvar,:lcon,:ucon]]
    all(bl .<= x .<= bu) && (nlp.meta.ncon == 0 || all(cl .<= cons(nlp, x) .<= cu))
  end

  n = nlp.meta.nvar

  x = copy(nlp.meta.x0)
  if !feasible(x)
    error("Starting point must be feasible")
  end
  fx = f(x)

  Δ = min(1.0, 0.1*minimum(nlp.meta.uvar - nlp.meta.lvar))

  iter = 0
  xt = zeros(n)
  ℓ = 0
  t = t₀ = Primes.PRIMES[n]
  q, _ = adjhalton(t₀, n, 0)
  smallestΔ = Δ
  maxt = t
  # Dk = [Hk -Hk]
  # Hk = qᵀq I - 2 qqᵀ
  # Hk eⱼ = qᵀq eⱼ - 2 qᵀeⱼ q = qᵀq eⱼ - 2qⱼ q

  if verbose
    println("Using $max_f f evaluations")
    @printf("%-5s  %-5s  %10s  %10s  %3s,%-3s  %10s  %s\n", "Feval", "Iter", "f(x)", "Δ", "t", "ℓ", "‖q‖", "status")
    @printf("%-5d  %5d  %10.4e  %10.4e  %3d,%-3d  %10.4e\n", neval_obj(nlp), 0, fx, Δ, t, ℓ, norm(q))
  end

  while !(Δ <= tol || neval_obj(nlp) >= max_f)
    decrease = false
    status = "No decrease"
    besti = 0
    bestf = fx
    for s = [1, -1]
      for i = 1:n
        xt .= x .- (2s * Δ * q[i]) .* q
        xt[i] += dot(q, q) * s * Δ
        if feasible(xt)
          ft = f(xt)
          if ft < bestf
            decrease = true
            status = "Decrease at i = $i, s = $s"
            besti = s * i
            bestf = ft
          end
        end
      end
    end
    if decrease
      fx = bestf
      i = abs(besti)
      s = sign(besti)
      x .-= (2s * Δ * q[i]) .* q
      x[i] += dot(q, q) * s * Δ
    end
    if verbose
      @printf("%-5d  %5d  %10.4e  %10.4e  %3d,%-3d  %10.4e  %s\n", neval_obj(nlp), iter, fx, Δ, t, ℓ, norm(q), status)
      println("x = $x")
    end

    if !decrease
      ℓ += 1
      Δ /= 4
    else
      ℓ -= 1
      Δ *= 4
    end
    iter += 1
    if Δ < smallestΔ
      smallestΔ = Δ
      t = t₀ + ℓ
    else
      t = 1 + maxt
    end
    maxt = max(maxt, t)
    q, _ = adjhalton(t, n, ℓ)
  end

  return x, fx, iter
end
