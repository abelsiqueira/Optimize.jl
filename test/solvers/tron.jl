using CUTEst

facts("Simple test") do
  context("No bounds") do
    x0 = [1.0; 2.0]
    f(x) = dot(x,x)/2

    nlp = ADNLPModel(f, x0, lvar=[-Inf; -Inf], uvar=[Inf; Inf])

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact x --> roughly(zeros(2), 1e-12)
    @fact fx --> roughly(0.0, 1e-12)
    @fact π --> roughly(0.0, 1e-12)
    @fact optimal --> true
  end

  context("Bounds") do
    x0 = [1.0; 2.0]
    f(x) = dot(x,x)/2
    l = [0.5; 0.25]
    u = [1.2; 1.5]

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact x --> roughly(l)
    @fact fx --> f(l)
    @fact π --> roughly(0.0)
    @fact optimal --> true
  end

  context("Rosenbrock") do
    x0 = [-1.2; 1.0]
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0)

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact x --> roughly([1.0;1.0], 1e-3)
    @fact fx --> roughly(1.0, 1e-5)
    @fact π --> roughly(0.0, 1e-3)
    @fact optimal --> true
  end

  context("Rosenbrock inactive bounds") do
    l = [0.5; 0.25]
    u = [1.2; 1.5]
    x0 = (l+u)/2
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact x --> roughly([1.0;1.0], 1e-3)
    @fact fx --> roughly(1.0, 1e-5)
    @fact π --> roughly(0.0, 1e-3)
    @fact optimal --> true
  end

  context("Rosenbrock active bounds") do
    l = [0.5; 0.25]
    u = [0.9; 1.5]
    x0 = (l+u)/2
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    sol = [0.9; 0.81]

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact x --> roughly(sol, 1e-3)
    @fact fx --> roughly(f(sol), 1e-5)
    @fact π --> roughly(0.0, 1e-3)
    @fact optimal --> true
  end
end

facts("Fixed variables") do
  context("One fixed") do
    x0 = ones(3)
    l = [1.0; 0.0; 0.0]
    u = [1.0; 2.0; 2.0]
    f(x) = 0.5*dot(x - 3, x - 3)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact optimal --> true
    @fact π --> roughly(0.0, 1e-3)
    @fact x --> roughly([1.0; 2.0; 2.0], 1e-3)
    @fact x[1] --> 1.0
  end

  context("All fixed") do
    n = 100
    x0 = zeros(n)
    l = 0.9*ones(n)
    u = copy(l)
    f(x) = sum(x.^4)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact optimal --> true
    @fact π --> 0.0
    @fact x --> l
  end
end

facts("Larger test") do
  n = 10
  Λ = linspace(1e-2, 1.0, n)
  context("Quadratic unconstrained") do
    for t = 1:10
      (Q,R) = qr(rand(n,n))
      r = ones(n)
      x0 = r + 1./Λ
      B = Q'*diagm(Λ)*Q
      f(x) = 1.0 + 0.5 * dot(x-r, B * (x-r))

      nlp = ADNLPModel(f, x0)

      x, fx, π, iter, optimal, tired, status = tron(nlp)
      @fact norm(B*(x-r)) --> roughly(0.0, 1e-5*n)
      @fact fx --> roughly(1.0, 1e-5)
      @fact π --> roughly(0.0, 1e-3)
      @fact optimal --> true
    end
  end

  context("Positive quadratic with bounds") do
    for t = 1:100
      (Q,R) = qr(rand(n,n))
      r = zeros(n)
      l, u, λl, λu = -1-rand(n), 1+rand(n), zeros(n), zeros(n)
      B = Q'*diagm(Λ)*Q
      for i = 1:n
        rnd = rand()
        if rnd < 1//3
          r[i] = l[i]
          λl[i] = 1 + rand()
        elseif rnd < 2//3
          r[i] = u[i]
          λu[i] = 1 + rand()
        else
          r[i] = l[i] + (0.1 + 0.8*rand())*(u[i] - l[i])
        end
      end
      v = -B*r - λu + λl
      x0 = [l[i] + rand() * (u[i] - l[i]) for i = 1:n]
      f(x) = 0.5 * dot(x, B*x) + dot(x, v)

      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

      x, fx, π, iter, optimal, tired, status = tron(nlp)

      @fact norm(B*(x-r)) --> roughly(0.0, 1e-5*n)
      @fact fx --> roughly(f(r), 1e-5)
      @fact π --> roughly(0.0, 1e-3)
      @fact optimal --> true
      @fact iter --> less_than_or_equal(10000)
      @fact status --> "first-order stationary"
    end
  end
end


facts("Problems") do
  n = 10
  x0 = 10*ones(n)
  f(x) = begin
    1e7*sum(exp(x))
  end

  nlp = ADNLPModel(f, x0)

  context("Iteration limit") do
    x, fx, π, iter, optimal, tired, status = tron(nlp, itmax=1)
    @fact iter --> 1
    @fact tired --> true
  end

  context("Time limit") do
    x, fx, π, iter, optimal, tired, status, el_time = tron(nlp, timemax=0)
    @fact el_time --> greater_than(0)
    @fact tired --> true
  end

  context("x0 outside box") do
    l = rand(n)
    u = l + rand(n)
    x0 = u + rand(n)
    f(x) = dot(x,x)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @fact x --> roughly(l, 1e-3)
    @fact fx --> roughly(f(l), 1e-5)
    @fact π --> roughly(0.0, 1e-3)
    @fact optimal --> true
  end
end

facts("Scaling") do
  n = 30
  function f(x)
    fx = 1.0
    for i = 1:n-1
      fx += 100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2
    end
    return fx
  end
  x0 = [i/(n+1) for i = 1:n]

  nlp = ADNLPModel(f, x0)

  x, fx, π, iter, optimal, tired, status = tron(nlp, timemax=600)

  @fact x --> roughly(ones(n), 1e-5*n)
  @fact fx --> roughly(1.0, 1e-3)
  @fact π --> roughly(0.0, 1e-3)
  @fact optimal --> true
end

facts("CUTEst") do
  problems = ["DENSCHNA", "DENSCHNB", "DENSCHNC", "DENSCHND", "DENSCHNE",
              "DENSCHNF", "BDEXP", "MCCORMCK"]
  @printf("%8s  %5s  %4s  %9s  %9s  %9s  %6s  %6s  %s\n", "Problem", "n",
      "type", "f(x)", "π", "time", "it", "nf", "status")
  for p in problems
    nlp = CUTEstModel(p)
    x, fx, π, iter, optimal, tired, status, el_time = tron(nlp, timemax=30)
    cutest_finalize(nlp)

    ctype = length(nlp.meta.ifree) == nlp.meta.nvar ? "unc" : "bnd"
    @printf("%8s  %5d  %4s  %9.2e  %9.2e  %9.2e  %6d  %6d  %s\n", p,
        nlp.meta.nvar, ctype, fx, π, el_time, iter, nlp.counters.neval_obj,
        status)
  end
end

