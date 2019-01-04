# Sthocastic Gradient Descent for individual functions in NLPModels
#
# min ∑σᵢfᵢ(x)
#

using NLPModels: nobjs
using Random, Statistics

export sgd

function trunk_tuning_problem()
  params = [:learning_rate]
  x0   = [   1.0]
  lvar = [1e-4]
  uvar = [ 1e4]
  c = nothing
  lcon = []
  ucon = []
  return params, x0, lvar, uvar, c, lcon, ucon
end

function sgd(nlp :: AbstractNLPModel;
             learning_rate :: Real=1e-1,
             keep_last :: Int=5,
             max_iter :: Int=10,
             max_f :: Int=0,
             max_time :: Float64=Inf)

  start_time = time()
  elapsed_time = 0.0

  nobjs = nlp.meta.nobjs
  if nobjs < 1
    error("sgd should be used for problems with more that one objective")
  end

  x = copy(nlp.meta.x0)
  n = nlp.meta.nvar
  X = zeros(n, keep_last)

  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))

  status = :unknown
  iter = 0
  tired = false
  I = collect(1:nobjs)

  k = 1
  X[:,1] .= x

  while !tired
    iter = iter + 1
    shuffle!(I)
    for i = I
      x = x - learning_rate * grad(nlp, i, x)
      k = k % keep_last + 1
      X[:,k] .= x
    end
    learning_rate *= 0.9

    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_f || elapsed_time > max_time || iter > max_iter
  end

  if tired
    if neval_obj(nlp) > max_f
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    elseif iter > max_iter
      status = :max_iter
    end
  end

  return GenericExecutionStats(status, nlp, solution=mean(X, dims=2)[:],
                               iter=iter, elapsed_time=elapsed_time)
end
