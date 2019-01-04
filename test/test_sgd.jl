using NLPModels, Optimize

function test_sgd()
  @testset "SGD works reasonbly on Linear Least Squares" begin
    m, n = 100, 5
    for t = 1:100
      A = rand(m, n)
      b = rand(m)
      fi(x,i) = (dot(A[i,:], x) - b[i])^2
      nlp = ADNLPModel([x -> fi(x,i) for i = 1:m], ones(m), zeros(n))
      stats = sgd(nlp, learning_rate=1e-2, max_iter=10)

      xsgd = stats.solution
      rsgd = b - A * xsgd

      xls = A \ b
      rls = b - A * xls

      @test norm(rsgd) ≤ norm(rls) * 1.05
    end
  end

  @testset "SGD on logistic regression" begin
    m, n = 100, 5
    for t = 1:100
      X = [ones(m) rand(m, n-1)]
      θ = randn(n)
      y = [dot(X[i,:], θ) > 0 ? 1 : 0 for i = 1:m]
      σ(t) = 1 / (1 + exp(-t))
      fi(θ,i) = y[i] == 0 ? -log(σ(dot(X[i,:],θ))) : -log(1 - σ(dot(X[i,:],θ)))
      nlp = ADNLPModel([θ -> fi(θ,i) for i = 1:m], ones(m), zeros(n))
      stats = sgd(nlp, learning_rate=1e-2, max_iter=10)

      θsgd = stats.solution
      ypred = [σ(dot(X[i,:], θ)) > 0.5 ? 1 : 0 for i = 1:m]
      acc = count(y .== ypred) / m
      @test acc == 1.0
    end
  end
end

test_sgd()
