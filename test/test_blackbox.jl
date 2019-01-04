function test_blackbox()
  @testset "Blackbox" begin
    @testset "Rosenbrock" begin
      f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
      nlp = ADNLPModel(f, [-1.2; 1.0])
      x, fx = blackbox(nlp, verbose=false, max_f=1_000_000, tol=1e-12)
      @test norm(x - ones(2)) < 1e-1
      @test fx < 1e-3
    end

    @testset "Bounded quadratic" begin
      f(x) = (x[1] - 1)^2 + 4 * (x[2] - 2)^2
      nlp = ADNLPModel(f, zeros(2), lvar=zeros(2), uvar=0.5*ones(2))
      x, fx = blackbox(nlp, verbose=false, max_f=1_000_000, tol=1e-6)
      @test x ≈ 0.5*ones(2)
    end

    @testset "Linearly constrained quadratic" begin
      f(x) = (x[1] - 1)^2 + 4 * (x[2] - 2)^2
      c(x) = x[1] + x[2] - 1.0
      nlp = ADNLPModel(f, zeros(2), c=c, lcon=[-Inf], ucon=[0.0])
      x, fx = blackbox(nlp, verbose=false, max_f=1_000_000, tol=1e-12)
      #@test norm(x - [-0.6; 1.6]) < 1e-6
      @test fx < f(zeros(2))
    end

    @testset "Linearly constrained quadratic with bounds" begin
      f(x) = (x[1] - 1)^2 + 4 * (x[2] - 2)^2
      c(x) = x[1] + x[2] - 1.0
      nlp = ADNLPModel(f, zeros(2), lvar=zeros(2), uvar=ones(2), c=c,
                           lcon=[-Inf], ucon=[0.0])
      x, fx = blackbox(nlp, verbose=false, max_f=1_000_000, tol=1e-12)
      @test norm(x - [0.0; 1.0]) < 1e-4
      @test fx < f(zeros(2))
    end

    @testset "Narrow" begin
      f(x) = dot(x, x)
      c(x) = x[2] - x[1]
      ϵ = 1e-4
      nlp = ADNLPModel(f, ones(2), c=c, lcon=[-ϵ], ucon=[ϵ])
      x, fx = blackbox(nlp, verbose=false, max_f=1_000_000, tol=1e-12)
      @test norm(x) < 1e-6
    end

    @testset "Bounded Rosenbrock" begin
      f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
      nlp = ADNLPModel(f, zeros(2), lvar=zeros(2), uvar=0.5*ones(2))
      x, fx = blackbox(nlp, verbose=false, max_f=1_000_000, tol=1e-12)
      @test norm(x - [0.5; 0.25]) < 1e-6
    end
  end
end

test_blackbox()
