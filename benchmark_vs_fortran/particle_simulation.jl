module ParticleSimulation

using StaticArrays
using Printf
using LinearAlgebra: norm

export Vec2D, random_point
export md, force_pair, wrap

function wrap(x,side)
    x = mod(x,side)
    if x >= side/2
        x -= side
    elseif x < -side/2
        x += side
    end
    return x
end        

function force_pair(x::T,y::T,cutoff,side) where T
    Δv = wrap.(y - x, side)
    d = norm(Δv)
    if d > cutoff
        return zero(T)
    else
        return 2*(d - cutoff)*(Δv/d)
    end
end

function forces!(f::Vector{T},x::Vector{T},force_pair::F) where {T,F}
    fill!(f,zero(T))
    n = length(x)
    for i in 1:n-1
        @inbounds for j in i+1:n
            fpair = force_pair(i,j,x[i],x[j])
            f[i] += fpair
            f[j] -= fpair
        end
    end
    return f
end

function md(x0,v0,mass,dt,nsteps,isave,force_pair::F) where F
    x = copy(x0)
    v = copy(v0)
    f = similar(x0)
    a = similar(x0)
    trajectory = [ copy(x0) ] # will store the trajectory
    for step in 1:nsteps
        # Compute forces
        forces!(f,x,force_pair)
        # Accelerations
        @. a = f / mass
        # Update positions
        @. x = x + v*dt + a*dt^2/2
        # Update velocities
        @. v = v + a*dt
        # Save if required
        if mod(step,isave) == 0
            push!(trajectory,copy(x))
        end
    end
    return trajectory
end

struct Vec2D{T} <: FieldVector{2,T}
    x::T
    y::T
end

function random_point(::Type{Vec2D{T}},range) where T 
    p = Vec2D(
        range[begin] + rand(T)*(range[end]-range[begin]),
        range[begin] + rand(T)*(range[end]-range[begin])
    )
    return p
end

end # module

using .ParticleSimulation

function main()
    n = 100
    cutoff = 5.
    side = 100.
    nsteps = 200_000
    trajectory = md((
        x0 = [random_point(Vec2D{Float64},(-50,50)) for _ in 1:n ], 
        v0 = [random_point(Vec2D{Float64},(-1,1)) for _ in 1:n ], 
        mass = [ 10.0 for _ in 1:100 ],
        dt = 0.001,
        nsteps = nsteps,
        isave = 1000,
        force_pair = (i,j,p1,p2) -> force_pair(p1,p2,cutoff,side)
    )...)
    file = open("traj_julia.xyz","w")
    for (step,x) in pairs(trajectory)
        println(file,n)
        println(file," step = ", step) 
        for p in x
            println(file,"He $(wrap(p.x,side)) $(wrap(p.y,side)) 0.")
        end
    end
    close(file)
end

main()

