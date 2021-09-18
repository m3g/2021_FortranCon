using StaticArrays
using CellListMap
using FastPow
using LinearAlgebra: norm
using Statistics: mean

struct Point3D{T} <: FieldVector{3,T}
	x::T
	y::T
	z::T
end

struct Point2D{T} <: FieldVector{2,T}
	x::T
	y::T
end

function random_point(::Type{Point3D{T}},range) where T 
	p = Point3D(
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin])
	)
	return p
end

function random_point(::Type{Point2D{T}},range) where T 
	p = Point2D(
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin])
	)
	return p
end

function energy(
	x,ε,σ,
	box::Box,cl::CellList,aux::CellListMap.AuxThreaded
)
	cl = UpdateCellList!(x,box,cl,aux)
	u = map_pairwise!(
        (xᵢ,xⱼ,i,j,d2,u) -> begin
			@fastpow u += ε*(σ^12/d2^6 - 2*σ^6/d2^3)
		end,
        zero(eltype(σ)), box, cl
    )
    return u
end

function forces!(
	f::Vector{T},
	x::Vector{T},
	ε,σ,
	box::Box,cl::CellList,aux::CellListMap.AuxThreaded
) where {T,F}
	cl = UpdateCellList!(x,box,cl,aux)
	fill!(f,zero(T))
	map_pairwise!(
		(xᵢ,xⱼ,i,j,d2,f) -> begin
			@fastpow ∂u∂xᵢ = 12*ε*(σ^12/d2^7 - σ^6/d2^4)*(xⱼ-xᵢ)
			@inbounds f[i] -= ∂u∂xᵢ
			@inbounds f[j] += ∂u∂xᵢ
			return f
		end,
		f, box, cl
	)
	return f
end

function packu(
	x,σ,
	box::Box,cl::CellList,aux::CellListMap.AuxThreaded
)
	cl = UpdateCellList!(x,box,cl,aux)
	u = map_pairwise!(
        (xᵢ,xⱼ,i,j,d2,u) -> begin 
			d = sqrt(d2)
			u += (d - σ)^2
			return u
		end,
        zero(eltype(σ)), box, cl
    )
    return u
end

function packg!(
	g::Vector{T},
	x::Vector{T},
	σ,
	box::Box,cl::CellList,aux::CellListMap.AuxThreaded
) where T
	cl = UpdateCellList!(x,box,cl,aux)
	fill!(g,zero(T))
	map_pairwise!(
		(xᵢ,xⱼ,i,j,d2,g) -> begin
			d = sqrt(d2)
			Δv = xⱼ - xᵢ 
			gₓ = -2*(d - σ)*(Δv/d)
			g[i] += gₓ
			g[j] -= gₓ
			return g
		end,
		g, box, cl
	)
	return g
end

function gradient_descent!(x::Vector{T},f,g!;tol=1e-3,maxtrial=500) where T
	gnorm(x) = maximum(norm(v) for v in x)
	itrial = 0
	step = 1.0
    xtrial = similar(x)
	g = fill!(similar(x),zero(T))
    fx = f(x)
    g = g!(g,x)
	while (gnorm(g) > tol) && (itrial < maxtrial) && (step > 1e-10)
		@. xtrial = x - step*g
		ftrial = f(xtrial)  
		@show itrial, step, fx, ftrial
		if ftrial >= fx
			step = step / 2
		else
            x .= xtrial
            fx = ftrial
			g = g!(g,x)
			step = step * 2
		end
		itrial += 1
	end 
	return x, gnorm(g), itrial
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

function run_md()

    n = 10_000
    side = 46
	ε = 0.0441795 # kcal/mol
    σ = 2*1.64009 # Å 
    mass_Ne = 20.17900 # g/mol 
	T0 = 300.
	kB = 0.001985875 # Boltzmann constant kcal / mol K
	cutoff = 12.

    x0 = [ random_point(Point3D{Float64},(-side/2,side/2)) for _ in 1:n ]
    #x0 = [ random_point(Point2D{Float64},(-side/2,side/2)) for _ in 1:n ]

    tol = 1.0
    box = Box([side,side,side],tol)
    #box = Box([side,side],tol)
    cl = CellList(x0,box)
    aux = CellListMap.AuxThreaded(cl)
    x, gnorm, itrial = gradient_descent!(
		copy(x0),
		(x) -> packu(x,tol,box,cl,aux),
		(g,x) -> packg!(g,x,tol,box,cl,aux)
	)

    box = Box([side,side,side],cutoff)
    #box = Box([side,side],cutoff)
    cl = CellList(x,box)
    aux = CellListMap.AuxThreaded(cl)
    x, gnorm, itrial = gradient_descent!(
		copy(x),
		(x) -> energy(x,ε,σ,box,cl,aux),
		(g,x) -> -forces!(g,x,ε,σ,box,cl,aux)
	)

	x0 = CellListMap.wrap_to_first.(x0,Ref(box))
	x = CellListMap.wrap_to_first.(x,Ref(box))

	# Initial velocities
	v0 = randn(Point3D{Float64},n)
	v_mean = mean(v0)
	@. v0 = v0 - Ref(v_mean)
	kinetic = (mass_Ne/2)*mean(v -> norm(v)^2, v0)
	@. v0 = v0 * sqrt(T0/(2*kinetic/(3*kB)))

    trajectory = md((
    	x0 = x0, 
    	v0 = v0, 
    	mass = [ mass_Ne for _ in 1:n ],
    	dt = 0.01,
    	nsteps = 1000,
    	isave = 10,
    	forces! = (f,x) -> forces!(f,x,ε,σ,box,cl,aux)
    )...)
    
    return trajectory
end

