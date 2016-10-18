Pkg.add("Compat")
using Compat
import Compat.String

const home = "https://github.com/JuliaSmoothOptimizers"
const deps = Dict{String, String}(
              "NLPModels" => "master",
              "OptimizationProblems" => "master",
              "Krylov" => "develop",
              "AmplNLReader" => "master",
              "BenchmarkProfiles" => "master")

const unix_deps = Dict{String, String}(
              "CUTEst" => "develop")

function dep_installed(dep)
  try
    println("Installed?")
    s = Pkg.installed(dep)  # throws an error instead of returning false
    println("Yes! >$s<")
    println(Pkg.installed())
    return true
  catch
    println("No!")
    return false
  end
end

function dep_install(dep, branch)
  println("Installing $dep")
  println(Pkg.installed())
  dep_installed(dep) || Pkg.clone("$home/$dep.jl.git")
  Pkg.checkout(dep, branch)
  Pkg.build(dep)
end

function deps_install(deps)
  for dep in keys(deps)
    println(Pkg.installed())
    dep_install(dep, deps[dep])
  end
end

println(Pkg.installed())
deps_install(deps)
@static if is_unix() deps_install(unix_deps); end
