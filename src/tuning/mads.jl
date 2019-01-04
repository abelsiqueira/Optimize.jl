using Primes

# t = ∑ᵣ aᵣ pʳ
function expansion(t :: Int, p :: Int)
  p > 1 || error("base $p not supported")
  t == 0 && return [0]
  E = Int[]
  while t > 0
    push!(E, t % p)
    t = div(t, p)
  end
  return E
end

function halton_utp(t :: Int, p :: Int)
  E = expansion(t, p)
  ne = length(E)
  return sum(E[i] / p^i for i = 1:ne)
end

# uₜ = (uₜₚ₁,…,uₜₚₙ)
# Starts at m-th prime
function halton(t :: Int, n :: Int)
  # Get n primes
  return [halton_utp(t, Primes.PRIMES[i]) for i = 1:n]
end

# qₜℓ
function adjhalton(t :: Int, n :: Int, ℓ :: Int)
  ut = halton(t, n)
  q = 2ut .- 1
  #α = sqrt(2^abs(ℓ)/n) - 0.5
  j = -1
  α = 0.0
  qt = q
  done = false
  isol = 0
  jsol = 0
  while !done
    j += 1
    done = true
    for i = 1:n
      ξ = (j + 0.5) * norm(q) / abs(q[i])
      qt = round.(Int, ξ * q / norm(q), RoundNearestTiesAway)
      qt[i] = (j + 1) * sign(q[i])
      if dot(qt, qt) <= 2^abs(ℓ)
        if α < ξ
          α = ξ
          isol = i
          jsol = j
        end
        done = false
      end
    end
  end
  i, j = isol, jsol
  α = (j + 0.5) * norm(q) / abs(q[i])
  qt = round.(Int, α * q / norm(q), RoundNearestTiesAway)
  qt[i] = (j + 1) * sign(q[i])
  return qt, α
end
