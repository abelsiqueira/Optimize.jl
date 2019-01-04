using Optimize, NLPModels, Random, LinearAlgebra, Logging, Printf

function confusion_matrix(yreal, ypred)
  α = [0; 1]
  return [count((yreal .== α[i]) .& (ypred .== α[j])) for i = 1:2, j = 1:2]
end

function logistic_regression(X, y; λ = 0.0)
  M, n = size(X)
  ϵ = sqrt(eps())
  σ(t) = 1 / (1 + exp(-t))
  E(θ) = begin
    Eθ = 0.0
    for i = 1:M
      hθ = σ(dot(X[i,:], θ))
      if y[i] == 1
        Eθ += log(hθ + ϵ)
      else
        Eθ += log(1 - hθ + ϵ)
      end
    end
    return -Eθ + λ * norm(θ)^2 / 2
  end

  logreg = ADNLPModel(E, zeros(n))
  stats = trunk(logreg)

  return stats.solution
end

function main()
  # Fake random data
  Random.seed!(0)
  disable_logging(Logging.Info)

  m = 1000
  n = 5
  X = rand(m, n) # ∈ [0,1]ⁿ
  X[:,3] = X[:,1] + X[:,2] + randn(m) * 0.01
  X = [ones(m) X]
  y = [sum(X[i,2:end].^2) + randn() * 2.5 > sqrt(n) ? 1 : 0 for i = 1:m]

  # Train set
  M = round(Int, 0.6m)
  X_tr, X_te = X[1:M,:], X[M+1:end,:]
  y_tr, y_te = y[1:M,:], y[M+1:end,:]

  σ(t) = 1 / (1 + exp(-t))

  for λ in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
    θ = logistic_regression(X_tr, y_tr, λ=λ)
    ypred = round.(Int, σ.(X_te * θ))
    cm = confusion_matrix(y_te, ypred)
    acc = tr(cm) / sum(cm)
    @printf("λ = %5.2e, acc = %6.4f\n", λ, acc)
  end

  function hyperparam(λv)
    θ = logistic_regression(X_tr, y_tr, λ=exp(λv[1]))
    ypred = round.(Int, σ.(X_te * θ))
    cm = confusion_matrix(y_te, ypred)
    acc = tr(cm) / sum(cm)
    return -acc
  end

  hypernlp = ADNLPModel(hyperparam, [0.0])
  bbsol = blackbox(hypernlp)
  λ = exp(bbsol[1][1])
  θ = logistic_regression(X_tr, y_tr, λ=λ)
  ypred = round.(Int, σ.(X_te * θ))
  cm = confusion_matrix(y_te, ypred)
  acc = tr(cm) / sum(cm)
  @printf("λ = %5.2e, acc = %6.4f\n", λ, acc)
end

main()
