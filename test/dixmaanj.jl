using NLPModels: increment!

mutable struct DIXMAANJ <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function DIXMAANJ(n :: Int=99)
  n % 3 == 0 || warn("dixmaanj: number of variables adjusted to be a multiple of 3")
  m = max(1, div(n, 3))
  n = 3m
  meta = NLPModelMeta(n, nobjs=1, nlsequ=0, llsrows=0, x0=2.0*ones(n))

  return DIXMAANJ(meta, Counters())
end

function NLPModels.obj(nlp :: DIXMAANJ, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  n = nvar(nlp)
  m = div(n, 3)
  α = 1.0
  β = γ = δ = 0.0625

  f = 1.0 + sum((i / n)^2 * α * x[i]^2                 for i = 1:n) +
            sum(β * x[i]^2 * (x[i + 1] + x[i + 1]^2)^2 for i = 1:n-1) +
            sum(γ * x[i]^2 * x[i + m]^4                for i = 1:2m) +
            sum((i / n)^2 * δ * x[i] * x[i + 2m]       for i = 1:m)
  return f
end

function NLPModels.grad!(nlp :: DIXMAANJ, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  n = nvar(nlp)
  m = div(n, 3)
  α = 1.0
  β = γ = δ = 0.0625
  for i = 1:n
    h = (i / n)^2
    g[i] = 2 * h * α * x[i]
    if i < n
      g[i] += 2β * x[i] * (x[i + 1] + x[i + 1]^2)^2
    end
    if i > 1
      g[i] += 2β * x[i - 1]^2 * (x[i] + x[i]^2) * (1 + 2 * x[i])
    end
    if i ≤ 2m
      g[i] += 2γ * x[i] * x[i + m]^4
    end
    if i > m
      g[i] += 4γ * x[i - m]^2 * x[i]^3
    end
    if i ≤ m
      g[i] += δ * h * x[i + 2m]
    end
    if i > 2m
      g[i] += δ * ((i - 2m) / n)^2 * x[i - 2m]
    end
  end
  return g
end

function NLPModels.grad(nlp :: DIXMAANJ, x :: AbstractVector)
  g = zeros(nvar(nlp))
  return grad!(nlp, x, g)
end

function NLPModels.hess(nlp :: DIXMAANJ, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  n = nvar(nlp)
  m = div(n, 3)
  nnz = 3n - 1
  α = 1.0
  β = γ = δ = 0.0625
  I = [1:n; 2:n; m+1:n; 2m+1:n]
  J = [1:n; 1:n-1; 1:2m; 1:m]
  V = zeros(nnz)
  for i = 1:n
    V[i] = 2 * (i / n)^2 * α
    if i < n
      V[i] += 2β * (x[i + 1] + x[i + 1]^2)^2
      V[n + i] = 4β * x[i] * (x[i + 1] + x[i + 1]^2) * (1 + 2 * x[i + 1])
    end
    if i > 1
      V[i] += 2β * x[i - 1]^2 * (1 + 6 * x[i] + 6 * x[i]^2)
    end
    if i ≤ 2m
      V[i] += 2γ * x[i + m]^4
      V[2n - 1 + i] = 8γ * x[i] * x[i + m]^3
    end
    if i ≤ m
      V[8m - 1 + i] = (i / n)^2 * δ
    else
      V[i] += 12γ * x[i - m]^2 * x[i]^2
    end
  end
  return sparse(I, J, obj_weight * V)
end

function NLPModels.hprod!(nlp :: DIXMAANJ, x :: AbstractVector, v :: AbstractVector,
                          Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  H = hess(nlp, x, obj_weight=obj_weight, y=y)
  Hv .= Symmetric(H, :L) * v
  return Hv
end

function NLPModels.hprod(nlp :: DIXMAANJ, x :: AbstractVector, v :: AbstractVector;
                         obj_weight=1.0, y=Float64[])
  Hv = zeros(nvar(nlp))
  return hprod!(nlp, x, v, Hx, obj_weight=obj_weight, y=y)
end
