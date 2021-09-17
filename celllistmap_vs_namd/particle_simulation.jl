using StaticArrays
import LinearAlgebra: norm
using CellListMap

struct Point3D{T} <: FieldVector{3,T}
	x::T
	y::T
	z::T
end

function random_point(::Type{Point3D{T}},range) where T 
	p = Point3D(
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin])
	)
	return p
end

function wrap(x,side)
	x = rem(x,side)
	if x >= side/2
		x -= side
	elseif x < -side/2
		x += side
	end
	return x
end

function force_pair!(x,y,i,j,d2,f,cutoff)
    Δv = y - x
    d = sqrt(d2)
	fpair = (Δv/d)*(d - cutoff)^2
    f[i] -= fpair
    f[j] += fpair
    return f
end

function gradient_descent(x,f,g,tol,maxtrial)
	itrial = 0
	step = 1.0
	xbest, fbest, grad = x0, f(x), g(x)
	fx = fbest
	while (abs(grad) > tol) && (itrial < maxtrial)
		xtrial = x - grad*step
		ftrial = f(xtrial)
		if ftrial > fx
			step = step / 2
		else
			x, fx = xtrial, ftrial
			grad = g(x)
			step = 1.0
			if fx < fbest
				xbest = x
				fbest = fx
			end
		end
		itrial += 1
	end 
	return x, grad, itrial
end

function md(
	x0::Vector{T},
	v0::Vector{T},
	mass,dt,nsteps,isave,forces!::F
) where {T,F}
	x = copy(x0)
	v = copy(v0)
	a = similar(x0)
	f = similar(x0)
	trajectory = [ copy(x0) ] # will store the trajectory
	for step in 1:nsteps
		# Compute forces
		forces!(f,x)
		# Accelerations
		@. a = f / mass
		# Update positions
		@. x = x + v*dt + a*dt^2/2
		# Update velocities
		@. v = v + a*dt
		# Save if required
		if mod(step,isave) == 0
			println("Saved trajectory at step: ",step)
			push!(trajectory,copy(x))
		end
	end
	return trajectory
end

function forces_fast!(
	f::Vector{T},
	x::Vector{T},
	force_pair::F,
	box::Box,cl::CellList,aux::CellListMap.AuxThreaded
) where {T,F}
	cl = UpdateCellList!(x,box,cl,aux)
	fill!(f,zero(T))
	map_pairwise!(force_pair, f, box, cl)
	return f
end

function run_md()

    n = 10_000
    side = 46
    cutoff = 5.

    x0 = [ random_point(Point3D{Float64},(-side/2,side/2)) for _ in 1:n ]
    v0 = [ random_point(Point3D{Float64},(-1,1)) for _ in 1:n ]

    box = Box([side,side,side],12)
    cl = CellList(x0,box)
    aux = CellListMap.AuxThreaded(cl)

    trajectory = md((
    	x0 = x0, 
    	v0 = v0, 
    	mass = [ 10.0 for _ in 1:n ],
    	dt = 0.1,
    	nsteps = 1000,
    	isave = 10,
    	forces! = (f,x) -> forces_fast!(
            f,x, 
            (p1,p2,i,j,d2,f) -> force_pair!(p1,p2,i,j,d2,f,cutoff),
            box, cl, aux
        )
    )...)
    
    return trajectory
end

