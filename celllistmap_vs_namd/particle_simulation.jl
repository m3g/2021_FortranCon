using StaticArrays
import LinearAlgebra: norm

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

function force_pair(x::T,y::T,cutoff,side) where T
	Δv = wrap.(y - x, side)
	d = norm(Δv)
	if d > cutoff
		return zero(T)
	else
		return (Δv/d)*(d - cutoff)^2
	end
end


function forces!(f::Vector{T},x::Vector{T},force_pair::F) where {T,F}
	fill!(f,zero(T))
	n = length(x)
	for i in 1:n-1
		for j in i+1:n
			fpair = force_pair(i,j,x[i],x[j])
			f[i] -= fpair
			f[j] += fpair
		end
	end
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

trajectory = md((
	x0 = [random_point(Point2D{Float64},(-50,50)) for _ in 1:100 ], 
	v0 = [random_point(Point2D{Float64},(-1,1)) for _ in 1:100 ], 
	mass = [ 1.0 for _ in 1:100 ],
	dt = 0.1,
	nsteps = 1000,
	isave = 10,
	forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> force_pair(p1,p2,cutoff))
)...)
