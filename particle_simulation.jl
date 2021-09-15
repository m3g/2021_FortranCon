### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ a82f22e4-f35b-461a-b481-1dff43722e44
using StaticArrays

# ╔═╡ d42f842d-6c2a-40db-b0c4-e936244a9e7c
using BenchmarkTools

# ╔═╡ d9bff7a0-7ce6-447b-ba76-120c691f6c0a
using Measurements

# ╔═╡ d6564250-f646-40de-9463-a956af1a5b1d
using ForwardDiff

# ╔═╡ 84e305e9-6ab9-4005-a295-a12c4eff68c5
using Optim

# ╔═╡ 7c792b6b-b6ee-4e30-88d5-d0b8064f2734
begin
	using Plots
	plot_font = "Computer Modern"
	default(
		fontfamily=plot_font,
		linewidth=2, framestyle=:box, label=nothing, grid=false,
		size=(400,400)
	)	
end

# ╔═╡ febe8c06-b3aa-4db1-a3ea-fdc2a81bdebd
using Printf

# ╔═╡ a756dd18-fac6-4527-944e-c16d8cc4bf95
begin
    using PlutoUI
    TableOfContents()
end

# ╔═╡ 4b484cf6-4888-4f04-b3fd-94862822b0c0
md"""
# Defining the type of particle *for the first time*

A simple point in 2D space, with coordinates `x` and `y`.
"""

# ╔═╡ 8c444ee4-8c77-413a-bbeb-9e5ae2428876
struct Point{T}
	x::T
	y::T
end

# ╔═╡ a0de01b5-779a-48c0-8d61-12b02a5f527e
md"""
We will define by hand the arithmetics needed for this kind of point. We could avoid doing all this by using the `StaticArrays` package, but for didactical reasons today we will write our arithmetics manually. Here we define what it means to add, substract, multiply and divide the points for the operations we will need. We also define a funtion that returns a point with null coordinates (the `zero` function), and a function that returns a random point in a desired interval. 

We will discuss these functions and why it may be interesting to define them manually later. 
"""

# ╔═╡ 414790ef-a592-418d-b116-9864b76530bf
begin
	import LinearAlgebra: norm
	import Base: -, +, *, /, zero
	-(p1::Point,p2::Point) = Point(p1.x - p2.x, p1.y - p2.y)
	+(p1::Point,p2::Point) = Point(p1.x + p2.x, p1.y + p2.y)
	*(x,p1::Point) = Point(x*p1.x, x*p1.y)
	*(p1::Point,x) = x*p1
	/(p1::Point,x) = Point(p1.x/x,p1.y/x)
	zero(::Type{Point{T}}) where T = Point(zero(T),zero(T))
	norm(p::Point) = sqrt(p.x^2 + p.y^2)
	function random_point(::Type{Point{T}},range) where T 
		p = Point(
			range[begin] + rand(T)*(range[end]-range[begin]),
			range[begin] + rand(T)*(range[end]-range[begin])
		)
		return p
	end
end

# ╔═╡ dc5f7484-3dc3-47a7-ad4a-30f97fc14d11
md"""
## Force between a pair of particles 

The function will be a soft potential, which is zero for distances greater than a cutoff, and increasing quadratically for distances smaller than the cutoff:

$$f(x,y,c)=
\begin{cases}
0\textrm{~if~}||x-y|| > c \\
(||x-y||-c)^2\textrm{~if~}||x-y||<c 
\end{cases}$$
"""

# ╔═╡ 0f52365d-34f4-46ed-923e-3ea31c6db0ca
function force_pair(x::T,y::T,cutoff) where T
	Δv = y - x
	d = norm(Δv)
	if d > cutoff
		return zero(T)
	else
		Δv = Δv / d
		return Δv*(d - cutoff)^2
	end
end

# ╔═╡ b5c09cd3-6063-4a36-96cd-2d128aa11b82
const cutoff = 5.

# ╔═╡ 7719b317-e85b-4583-b401-a8614d4b2373
md"""
The function that will compute the force over all pairs will just *naively* run over all the pairs. The function `forces!` will receive as a parameter the function that computes the force between pairs, to allow for its generality.

Inside `forces!`, the `force_pair` function will receive four parameters: the indexes of the particles and their positions. We will use the indexes later. 
"""

# ╔═╡ f58769a6-a656-42a3-8bc6-c204d4cfd897
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

# ╔═╡ 144119ad-ab88-4165-883a-f2fc2464a838
md"""
Let us create some points to explain how the function will be called. 
"""

# ╔═╡ dd9e4332-2908-40ef-b461-6b571df56cf4
md"""
The function `force_pair`, will be passed to the function that computes the forces to all pairs as *closure*, which will capture the value of the cutoff. The closure also allows us to ignore the indexes of the particles, which are expected by the inner implementation of the function inside `forces`. For example:
"""

# ╔═╡ 0d964841-7764-48a4-9a6d-0b017ce4a90e
v = [ Point(1.,1.), Point(2.,2.) ]

# ╔═╡ c56e1910-facc-4595-81e8-e2d5d8c4e8f4
v[1] = Point(0.,0.)

# ╔═╡ d54598a0-1190-402c-8b51-2d09ca47cdf0
v

# ╔═╡ b5206dd5-1f46-4437-929b-efd68393b12b
md"""
# Performing a particle simulation

Now, given the function that computes the forces, we can perform a particle simulation. We will use a simple Euler integration scheme, and the algorithm will be:

1. Compute forces at time $t$ from positions $x$:
$f(t) = f(x)$

2. Update the positions (using $a = f/m$ for $m=1$):
$x(t + dt) = x(t) + v(t)dt + a(t)dt^2/2$

3. Update the velocities:
$v(t+dt) = v(t) + a(t)*dt$

4. Goto 1.

## The actual simulation code is as short:
"""

# ╔═╡ eb5dc224-1491-11ec-1cae-d51c93cd292c
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

# ╔═╡ 66c7d960-7e05-4613-84e8-2a40fe40dc3d
md"""
## Lets run the simulation!
"""

# ╔═╡ e717a8d9-ccfb-4f89-b2a2-f244f108b48d
md"""
Here we generate random positions and velocities, and use masses equal to `1.0` for all particles.
"""

# ╔═╡ eab7b195-64d5-4587-8687-48a673ab091b
md"""
# Using periodic boundary conditions

Our particles just explode, since they have initial random velocities and there are only repulsive interactions. 

We have a more interesting dynamics if we use periodic boundary conditions. To do so, we will update how the forces are computed.

We need all arithmetics of the points and some other point operations, like how to iterate over the point coordinates. Let us save our time using an implementation of that:
"""

# ╔═╡ 27c0e8f3-dc17-46ae-a429-34cb453df888
struct Point2D{T} <: FieldVector{2,T}
	x::T
	y::T
end

# ╔═╡ ed35d044-8506-4ec0-a2d3-03202d0c29a5
md"""
The `Point2D` structure above, by being a subtype of `FieldVector` from `StaticArrays`, has already all the arithmetics defined. 
"""

# ╔═╡ 101ac577-6f2f-41a7-852f-d1de22c597e3
md"""
We only need to define our custom random point generator:
"""

# ╔═╡ 8914ae52-7f09-483a-8ca9-15530aadd371
function random_point(::Type{Point2D{T}},range) where T 
	p = Point2D(
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin])
	)
	return p
end

# ╔═╡ 367a686c-4cab-4f13-b285-c3243168cfb1
md"""
And steal the `norm` function from `LinearAlgebra`, which cannot be directly used here just because it conflicts with our previous definition of `norm`.
"""

# ╔═╡ 34dc72dc-4864-47c0-b730-183f67e7aea3
md"""
## Wrapping of coordinates

The following function defines how to wrap the coordinates on the boundaries, for a square or cubic box of side `side`:
"""

# ╔═╡ beeb3335-5c49-47de-a1d3-3eef5f9479f1
function wrap(x,side)
	x = rem(x,side)
	if x >= side/2
		x -= side
	elseif x < -side/2
		x += side
	end
	return x
end		

# ╔═╡ 02d9bf3b-708c-4293-b198-9043b334ff7e
md"""
This allows writting the force computation now as:
"""

# ╔═╡ 0967b90d-ac88-476d-a57a-7c38dfa82204
function force_pair(x::T,y::T,cutoff,side) where T
	Δv = wrap.(y - x, side)
	d = norm(Δv)
	if d > cutoff
		return zero(T)
	else
		Δv = Δv / d
		return Δv*(d - cutoff)^2
	end
end

# ╔═╡ 0a5282ee-c88a-4bcc-aca2-477f28e9e04d
md"""
Our box has a side of 100:
"""

# ╔═╡ b2a4a505-47ff-40bb-9a6d-a08d91c53217
const side = 100.

# ╔═╡ fcff6973-012a-40fc-a618-f6262266287a
md"""
To run the simulation with the new periodic forces, we use the same `md` function, just passing the new `force_pair` function:
"""

# ╔═╡ c4798182-de75-4b59-8be7-f7cf1051364d
md"""
While plotting the trajectory, we will wrap the coordinates:
"""

# ╔═╡ 22fb0386-d4fa-47b9-ac31-decf2731cbc1
md"""
## Benchmarking
"""

# ╔═╡ 2a3e7257-63ad-4761-beda-cec18b91f99c
md"""

Something of the order of `200ms` and `187KiB` of allocations does not seem bad, but it doesn't mean anything either. What is interesting to point here is just that this code, compared to ahead-of-time compiled language like Fortran, is completely comparable in terms of performance, as [this benchmark](https://github.com/m3g/2021_FortranCon/tree/main/benchmark_vs_fortran) shows. 

"""

# ╔═╡ 49b1f040-929a-4238-acd9-6554757b592c
md"""
# And now the fun begins

Now we will show the generic code flexibility.

## Running the simulations in 3D

Not much is needed to just run the simulation in three dimensions. We only need to define our 3D point:
"""

# ╔═╡ 26d5c6e9-a903-4792-a0e0-dec1a2e86a01
struct Point3D{T} <: FieldVector{3,T}
	x::T
	y::T
	z::T
end

# ╔═╡ 1ce41841-1ca7-43e4-a08a-21142e29ed93
function random_point(::Type{Point3D{T}},range) where T 
	p = Point3D(
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin]),
		range[begin] + rand(T)*(range[end]-range[begin])
	)
	return p
end

# ╔═╡ 2aef27bd-dea6-4a93-9d0f-b9249c9dd2cd
md"""
That is enough such that we can run the simulations in 3D:
"""

# ╔═╡ b6dcb9a3-59e3-4eae-9399-fb072c704f1a
md"""
## Automatic error propagation

Performing simulations in different dimensions is not the most interesting, or most useful property of generic programming. We can, more interestingly, propagate the error in the positions of the particles, simply by defining a type of particle that carries both the position and the cumulative error. 

The concept is identical to what we have done initialy when we defined `Point` and its arithmetics. We would need to define a new type of structure carrying the positions and the errors, and define the appropriate error propagataion arithmetics for the operations that will be performed with that type of variable. 

Fortunately, again, this is done, in the `Measurements` package:
"""

# ╔═╡ 8267220a-f06e-4761-b310-00f8ba44e4b1
md"""
using `Measurments`  we do not need to change anything in our previous code, but only redefine the content of our points, which will now carry in each coordinate the position and the error in the position, accumulated from an initial uncertainty:
"""

# ╔═╡ a0e39d66-e328-4b61-86d4-99df6b832b7a
p = Point2D( rand() ± 1e-5, rand() ± 1e-5 )

# ╔═╡ 418f31bb-81d5-459b-b402-4fd4e3f4ab27
md"""
We need to redefine your initial random point generator only:
"""

# ╔═╡ 05402cbd-78c6-4234-8680-c351c8c37778
function random_point(::Type{Point2D{Measurement{T}}},range,uncertainty) where T	
	p = Point2D(
		range[begin] + rand(T)(range[end]-range[begin]) ± uncertainty*rand(),
		range[begin] + rand(T)(range[end]-range[begin]) ± uncertainty*rand()
	)
	return p
end

# ╔═╡ ecdaaca2-f5d3-496c-960f-df9578268023
x0 = [ random_point(Point{Float64},-50:50) for _ in 1:100 ]

# ╔═╡ a2226893-4f32-4ec3-aaef-1c304467452c
f = similar(x0)

# ╔═╡ 5ad2af1d-5c41-40d8-a451-fd99d9faafc2
forces!(f, x0, (i,j,p1,p2) -> force_pair(p1,p2,cutoff))

# ╔═╡ 3755a4f3-1842-4de2-965e-d294c06c54c7
trajectory = md((
	x0 = [random_point(Point2D{Float64},-50:50) for _ in 1:100 ], 
	v0 = [random_point(Point2D{Float64},-10:10) for _ in 1:100 ], 
	mass = [ 1.0 for _ in 1:100 ],
	dt = 0.1,
	nsteps = 1000,
	isave = 10,
	force_pair = (i,j,p1,p2) -> force_pair(p1,p2,cutoff,side)
)...)

# ╔═╡ 985b4ffb-7964-4b50-8c2f-e5f45f352500
trajectory_periodic = md((
	x0 = [random_point(Point2D{Float64},(-50,50)) for _ in 1:100 ], 
	v0 = [random_point(Point2D{Float64},(-0.1,0.1)) for _ in 1:100 ], 
	mass = [ 1.0 for _ in 1:100 ],
	dt = 0.01,
	nsteps = 10000,
	isave = 100,
	force_pair = (i,j,p1,p2) -> force_pair(p1,p2,cutoff,side)
)...)

# ╔═╡ 1ad401b5-20b2-489b-b2aa-92f729b1d725
@benchmark md($(
	x0 = [random_point(Point2D{Float64},-50:50) for _ in 1:100 ], 
	v0 = [random_point(Point2D{Float64},-1:1) for _ in 1:100 ], 
	mass = [ 1.0 for _ in 1:100 ],
	dt = 0.1,
	nsteps = 1000,
	isave = 10,
	force_pair = (i,j,p1,p2) -> force_pair(p1,p2,cutoff,side)
)...)

# ╔═╡ 0546ee2d-b62d-4c7a-8172-ba87b3c1aea4
trajectory_periodic_3D = md((
	x0 = [random_point(Point3D{Float64},-50:50) for _ in 1:100 ], 
	v0 = [random_point(Point3D{Float64},-1:1) for _ in 1:100 ], 
	mass = [ 1.0 for _ in 1:100 ],
	dt = 0.1,
	nsteps = 1000,
	isave = 10,
	force_pair = (i,j,p1,p2) -> force_pair(p1,p2,cutoff,side)
)...)

# ╔═╡ 4e97f24c-c237-4117-bc57-e4e88c8fb8d2
md"""
Which generates random points carrying an initial uncertainty we defined:
"""

# ╔═╡ b31da90d-7165-42de-b18d-90584affea03
random_point(Point2D{Measurement{Float64}},(-50,50),1e-5)

# ╔═╡ 1d6eedfd-d013-4557-9cf2-103f8fb7b72a
md"""
The trajectory, of course, looks the same:
"""

# ╔═╡ c003a61d-a434-4d7b-9214-5b52aa044248
md"""
But now we have an estimate of the error of the positions, propagated from the initial uncertainty:
"""

# ╔═╡ 63eb391f-0238-434a-bc3a-2fa8ed41448e
md"""
Perhaps this is more interesting to see in a planetary trajectory:
"""

# ╔═╡ 7b9bb0fd-34a5-42e1-bc35-7259447b73d0
function gravitational_force(i,j,x,y,mass)
	G = 0.00049823382528 # MKm³ / (10²⁴kg days²)
	dr = y - x
	r = norm(dr)
	return -G*mass[i]*mass[j]*dr/r^3
end

# ╔═╡ 6a4e0e2e-75c5-4cab-987d-3d6b62f9bb06
md"""
Note that now we need the indexes of the particles to be able to pass the information of their masses. 

A set of planetary positions and velocities is something that we have to obtain [experimentaly](https://nssdc.gsfc.nasa.gov/planetary/factsheet/). Here, the distance units $10^6$ km), and time is in days. Thus, velocities are in MKm per day.

The uncertainty of the positions will be taken as the diameter of each planet. In this illustrative example we will not add uncertainties to the velcities. 
"""

# ╔═╡ c91862dd-498a-4712-8e3d-b77e088cd470
planets_x0 = [
	Point2D(measurement(  0.0,   1.39), measurement(0.,   1.39)), # "Sun"
	Point2D(measurement( 57.9,  4.879e-3), measurement(0.,  4.879e-3)), # "Mercury"
	Point2D(measurement(108.2, 12.104e-3), measurement(0., 12.104e-3)), # "Venus"
	Point2D(measurement(149.6, 12.756e-3), measurement(0., 12.756e-3)), # "Earth"
	Point2D(measurement(227.9,  6.792e-3), measurement(0.,  6.792e-3)), # "Mars"
]

# ╔═╡ a08d6e6d-ddc4-40aa-b7c4-93ea03191415
planets_v0 = [ 
	Point2D(measurement(0., 0.), measurement(  0.0,   0.)), # "Sun"
	Point2D(measurement(0., 0.), measurement( 4.10,   0.)), # "Mercury"
	Point2D(measurement(0., 0.), measurement( 3.02,   0.)), # "Venus"
	Point2D(measurement(0., 0.), measurement( 2.57,   0.)), # "Earth"
	Point2D(measurement(0., 0.), measurement( 2.08,   0.)), # "Mars"	
]

# ╔═╡ a356e2cc-1cb1-457a-986c-998cf1efe008
md"""
And the masses are given in units of $10^{24}$ kg:
"""

# ╔═╡ 57141f7c-9261-4dc5-98e4-b136a15f86fc
const masses = [ 1.99e6, 0.330, 4.87, 5.97, 0.642 ]

# ╔═╡ 055e32d7-073c-40db-a267-750636b9f786
md"""
Let us see the planets orbiting the sun:
"""

# ╔═╡ aaa97ce4-a5ff-4332-89a2-843cee2e5b6d
trajectory_planets = md((
	x0 = planets_x0, 
	v0 = planets_v0, 
	mass = masses,
	dt = 1, # days
	nsteps = 2*365, # two earth years
	isave = 1, # save every day
	force_pair = (i,j,p1,p2) -> gravitational_force(i,j,p1,p2,masses)
)...)

# ╔═╡ c4344e64-aa22-4328-a97a-71e44bcd289f
md"""
One thing I don't like, though, is that in two years the earth appeared to have made much less than to complete revolutions around the sun. Something is wrong with our data. Can we improve that?
"""

# ╔═╡ 827bda6f-87d4-4d36-8d89-f144f4595240
md"""
## We can differentiate everything!

Perhaps astoningshly (at least for me), our simulation is completely differentiable. That means that we can tune the parameters of the simulation, and the data, using optimization algorithms that require derivatives. 

Here we speculate that what was wrong with our data was that the initial position of the earth was somewhat out of place. That caused the earth orbit to be slower than it should.

We will define, then, an objective function which returns the displacement of the earth relative to its initial position (at day one) after one year. Our goal is that after one year the earth returns to its initial position.
"""

# ╔═╡ 0103b69a-2505-42f8-8df4-d08759eba077
function error_in_orbit(x::T=149.6) where T
	x0 = [
		Point2D( zero(T), zero(T)), # "Sun"
		Point2D(      x,  zero(T))  # "Earth"
	]
	v0 = [ 
		Point2D( zero(T),     zero(T)), # "Sun"
		Point2D( zero(T), 2.57*one(T))  # "Earth"
	]
	masses = [ 1.99e6*one(T), 5.97*one(T) ]
	last_position = md((
		x0 = x0, 
		v0 = v0, 
		mass = masses,
		dt = 1, # days
		nsteps = 365, # one earth year
		isave = 365, # save only last point
		force_pair = (i,j,p1,p2) -> gravitational_force(i,j,p1,p2,masses)
	)...)
	return norm(last_position[end][2] - x0[2])
end

# ╔═╡ fda6171c-9675-4f2e-b226-7ccf100529cd
error_in_orbit()

# ╔═╡ 107aec28-ecb5-4007-95e5-25d0a7f0c465
ForwardDiff.derivative(error_in_orbit,149.6)

# ╔═╡ b8320f78-323c-49a9-a9f9-2748d19ecb35
error_derivative(x) = ForwardDiff.derivative(error_in_orbit,x)

# ╔═╡ 535716e6-9c1c-4324-a4cd-b1214df3c01d
function gradient_descent(x,f,g,tol,maxtrial)
	itrial = 0
	step = 1.0
	xbest = x0
	fbest = f(x)
	grad = g(x)
	while (abs(grad) > tol) && (itrial < maxtrial)
		xtrial = x - grad*step
		if f(xtrial) > f(x)
			step = step / 2
		else
			x = xtrial
			grad = g(x)
			step = 1.0
			if f(x) < fbest
				xbest = x
				fbest = f(x)
			end
		end
		itrial += 1
	end 
	return x, grad, itrial
end

# ╔═╡ 931a9c5f-8f91-4e88-956b-50c0efc9c58b
best_x0 = gradient_descent(149.6,error_in_orbit,error_derivative,1e-4,1000)

# ╔═╡ 7658a32c-d3da-4ec9-9d96-0d30bb18f08c
error_in_orbit(best_x0[1])

# ╔═╡ e61981d5-5448-45e9-81dc-320ac87ba813
md"""
Seems that it worked! Let us see our trajectory now with the new initial condition:
"""

# ╔═╡ 4a75498d-8f4e-406f-8b01-f6a5f153919f
function earth_trajectory(x::T) where T
	x0 = [
		Point2D( 0., 0.), # "Sun"
		Point2D(  x, 0.)  # "Earth"
	]
	v0 = [ 
		Point2D( 0., 0.  ), # "Sun"
		Point2D( 0., 2.57), # "Earth"
	]
	masses = [ 1.99e6, 5.97 ]
	earth_trajectory = md((
		x0 = x0, 
		v0 = v0, 
		mass = masses,
		dt = 1, # days
		nsteps = 365, # one earth year
		isave = 1, # save only last point
		force_pair = (i,j,p1,p2) -> gravitational_force(i,j,p1,p2,masses)
	)...)
	return earth_trajectory
end

# ╔═╡ 31e1bb51-c531-4c4a-8634-5caafb7e9e51
earth_traj_0 = earth_trajectory(149.6)

# ╔═╡ b0b81da4-6788-45c4-b618-188a02b5e09c
earth_traj_best = earth_trajectory(best_x0[1])

# ╔═╡ 8e618602-0c65-448f-adae-2c80e7cdd73e
earth_traj_0[end][2], earth_traj_best[end][2]

# ╔═╡ 2871aca3-e6b4-4a2d-868a-36562e9a274c
md"""
# Some notebook options and setup
"""

# ╔═╡ 2a2e9155-1c77-46fd-8502-8431573f94d0
md"""
## Default plot setup
"""

# ╔═╡ a9981931-4cc9-4d16-a6d2-34b4071a84d7
const build_plots = true

# ╔═╡ 374f239b-6470-40ed-b068-a8ecaace4f09
build_plots && plot(
	0:0.1:1.2*cutoff,force_pair.(0.,0:0.1:1.2*cutoff,cutoff),
	xlabel="Distance",ylabel="Force"
)

# ╔═╡ 43e6b146-ee35-40f1-b540-3da22b9e1b1b
build_plots && scatter([(x.x, x.y) for x in x0])

# ╔═╡ 505ef5ab-f131-4ab3-a723-795b5eb5dc0f
build_plots && @gif for x in trajectory
  	scatter([ (p.x,p.y) for p in x ], lims=(-1000,1000))
end

# ╔═╡ efc586a2-0946-4dc5-ab3a-3902a811f3ad
build_plots && @gif for x in trajectory_periodic
  	scatter([ wrap.((p.x,p.y),100) for p in x ], lims=(-60,60))
end

# ╔═╡ 4a498c18-406f-4437-b378-aa9fdc75b919
build_plots && @gif for x in trajectory_periodic_3D
  	scatter([ wrap.((p.x,p.y,p.z),100) for p in x ], lims=(-60,60))
end

# ╔═╡ d87c22d1-d595-4d43-ab1c-f28d282a3485
build_plots && ( trajectory_2D_error = md((
	x0 = [random_point(Point2D{Measurement{Float64}},(-50,50),1e-5) for _ in 1:100 ], 
	v0 = [random_point(Point2D{Measurement{Float64}},(-1,1),1e-5) for _ in 1:100 ],
	mass = [ 1.0 for _ in 1:100 ],
	dt = 0.1,
	nsteps = 1000,
	isave = 10,
	force_pair = (i,j,p1,p2) -> force_pair(p1,p2,cutoff,side)
)...) )

# ╔═╡ bf0a5303-f5ce-4711-b9ee-a12ce2d8a397
build_plots && @gif for x in trajectory_2D_error
  	positions = [ wrap.((p.x.val,p.y.val),100) for p in x ]
	scatter(positions, lims=(-60,60))
end

# ╔═╡ e24ce081-e367-4feb-8a79-66b8654a0b3a
build_plots && @gif for x in trajectory_2D_error
	histogram(
		[ p.x.err for p in x ],
		xlabel="Uncertainty in x",ylabel="Number of points",
		bins=0:1e-4:20e-4,ylims=[0,50]
	)
end

# ╔═╡ 1067527e-76b7-4331-b3ab-efd72fb99dfc
build_plots && @gif for (step,x) in pairs(trajectory_planets)
	colors = [ :yellow, :grey, :brown, :blue, :red ]
  	positions = [ (p.x.val,p.y.val) for p in x ]
	xerr = [ p.x.err for p in x ]
	yerr = [ p.y.err for p in x ] 
	scatter(positions,lims=[-250,250], color=colors, xerror=xerr, yerror=yerr)
	annotate!(150,-210,text(@sprintf("%5i days",step),plot_font,12))
end

# ╔═╡ 4cef9cea-1e84-42b9-bff6-b9a8b3bfe8da
build_plots && @gif for step in eachindex(earth_traj_best)
	colors = [ :yellow, :blue ]
  	positions0 = [ (p.x,p.y) for p in earth_traj_0[step] ] 
	positions_best = [ (p.x,p.y) for p in earth_traj_best[step] ]
	scatter(positions0,lims=[-250,250], color=colors, alpha=0.5)
	scatter!(positions_best,lims=[-250,250], color=colors)
	scatter!(
		(earth_traj_best[1][2].x,earth_traj_best[1][2].y),
		color=:white,alpha=0.5,
		markersize=10
	)
	annotate!(150,-210,text(@sprintf("%5i days",step),plot_font,12))
end

# ╔═╡ 043d89a4-ac5d-49ac-9820-b35c5ee967bc
begin 
	reset_element(x::Measurement{T}) where T = zero(T) ± x.err
	reset_element(x) = zero(x)
end

# ╔═╡ b5008faf-fd43-45dd-a5a1-7f51e0b4ede5
md"""
## Table of Contents
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
BenchmarkTools = "~1.1.4"
ForwardDiff = "~0.10.19"
Measurements = "~2.6.0"
Optim = "~1.4.1"
Plots = "~1.21.3"
PlutoUI = "~0.7.9"
StaticArrays = "~1.2.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "d84c956c4c0548b4caf0e4e96cf5b6494b5b1529"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.32"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "42ac5e523869a84eac9669eaceed9e4aa0e1587b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.4"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4ce9393e871aca86cc457d9f66976c3da6902ea7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "caf289224e622f518c9dbfe832cdafa17d7c80a6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.4"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "RecipesBase", "Requires"]
git-tree-sha1 = "31c8c0569b914111c94dd31149265ed47c238c5b"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.6.0"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "144bab5b1443545bc4e791536c9f1eacb4eed06a"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7863df65dbb2a0fa8f85fcaf0a41167640d2ebed"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.4.1"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9ff1c70190c1c30aebca35dc489f7411b256cd23"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.13"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "2dbafeadadcf7dadff20cd60046bba416b4912be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.21.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "d4491becdc53580c6dadb0f6249f90caae888554"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.0"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "f41020e84127781af49fc12b7e92becd7f5dd0ba"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.2"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "368d04a820fe069f9080ff1b432147a6203c3c89"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─4b484cf6-4888-4f04-b3fd-94862822b0c0
# ╠═8c444ee4-8c77-413a-bbeb-9e5ae2428876
# ╟─a0de01b5-779a-48c0-8d61-12b02a5f527e
# ╠═414790ef-a592-418d-b116-9864b76530bf
# ╟─dc5f7484-3dc3-47a7-ad4a-30f97fc14d11
# ╠═0f52365d-34f4-46ed-923e-3ea31c6db0ca
# ╠═b5c09cd3-6063-4a36-96cd-2d128aa11b82
# ╠═374f239b-6470-40ed-b068-a8ecaace4f09
# ╟─7719b317-e85b-4583-b401-a8614d4b2373
# ╠═f58769a6-a656-42a3-8bc6-c204d4cfd897
# ╟─144119ad-ab88-4165-883a-f2fc2464a838
# ╠═ecdaaca2-f5d3-496c-960f-df9578268023
# ╠═43e6b146-ee35-40f1-b540-3da22b9e1b1b
# ╠═a2226893-4f32-4ec3-aaef-1c304467452c
# ╟─dd9e4332-2908-40ef-b461-6b571df56cf4
# ╠═5ad2af1d-5c41-40d8-a451-fd99d9faafc2
# ╠═0d964841-7764-48a4-9a6d-0b017ce4a90e
# ╠═c56e1910-facc-4595-81e8-e2d5d8c4e8f4
# ╠═d54598a0-1190-402c-8b51-2d09ca47cdf0
# ╟─b5206dd5-1f46-4437-929b-efd68393b12b
# ╠═eb5dc224-1491-11ec-1cae-d51c93cd292c
# ╟─66c7d960-7e05-4613-84e8-2a40fe40dc3d
# ╟─e717a8d9-ccfb-4f89-b2a2-f244f108b48d
# ╠═3755a4f3-1842-4de2-965e-d294c06c54c7
# ╠═505ef5ab-f131-4ab3-a723-795b5eb5dc0f
# ╟─eab7b195-64d5-4587-8687-48a673ab091b
# ╠═a82f22e4-f35b-461a-b481-1dff43722e44
# ╠═27c0e8f3-dc17-46ae-a429-34cb453df888
# ╟─ed35d044-8506-4ec0-a2d3-03202d0c29a5
# ╟─101ac577-6f2f-41a7-852f-d1de22c597e3
# ╠═8914ae52-7f09-483a-8ca9-15530aadd371
# ╟─367a686c-4cab-4f13-b285-c3243168cfb1
# ╟─34dc72dc-4864-47c0-b730-183f67e7aea3
# ╠═beeb3335-5c49-47de-a1d3-3eef5f9479f1
# ╟─02d9bf3b-708c-4293-b198-9043b334ff7e
# ╠═0967b90d-ac88-476d-a57a-7c38dfa82204
# ╟─0a5282ee-c88a-4bcc-aca2-477f28e9e04d
# ╠═b2a4a505-47ff-40bb-9a6d-a08d91c53217
# ╟─fcff6973-012a-40fc-a618-f6262266287a
# ╠═985b4ffb-7964-4b50-8c2f-e5f45f352500
# ╟─c4798182-de75-4b59-8be7-f7cf1051364d
# ╠═efc586a2-0946-4dc5-ab3a-3902a811f3ad
# ╟─22fb0386-d4fa-47b9-ac31-decf2731cbc1
# ╠═d42f842d-6c2a-40db-b0c4-e936244a9e7c
# ╠═1ad401b5-20b2-489b-b2aa-92f729b1d725
# ╟─2a3e7257-63ad-4761-beda-cec18b91f99c
# ╟─49b1f040-929a-4238-acd9-6554757b592c
# ╠═26d5c6e9-a903-4792-a0e0-dec1a2e86a01
# ╠═1ce41841-1ca7-43e4-a08a-21142e29ed93
# ╟─2aef27bd-dea6-4a93-9d0f-b9249c9dd2cd
# ╠═0546ee2d-b62d-4c7a-8172-ba87b3c1aea4
# ╠═4a498c18-406f-4437-b378-aa9fdc75b919
# ╟─b6dcb9a3-59e3-4eae-9399-fb072c704f1a
# ╠═d9bff7a0-7ce6-447b-ba76-120c691f6c0a
# ╟─8267220a-f06e-4761-b310-00f8ba44e4b1
# ╠═a0e39d66-e328-4b61-86d4-99df6b832b7a
# ╟─418f31bb-81d5-459b-b402-4fd4e3f4ab27
# ╠═05402cbd-78c6-4234-8680-c351c8c37778
# ╟─4e97f24c-c237-4117-bc57-e4e88c8fb8d2
# ╠═b31da90d-7165-42de-b18d-90584affea03
# ╠═d87c22d1-d595-4d43-ab1c-f28d282a3485
# ╟─1d6eedfd-d013-4557-9cf2-103f8fb7b72a
# ╠═bf0a5303-f5ce-4711-b9ee-a12ce2d8a397
# ╟─c003a61d-a434-4d7b-9214-5b52aa044248
# ╠═e24ce081-e367-4feb-8a79-66b8654a0b3a
# ╟─63eb391f-0238-434a-bc3a-2fa8ed41448e
# ╠═7b9bb0fd-34a5-42e1-bc35-7259447b73d0
# ╟─6a4e0e2e-75c5-4cab-987d-3d6b62f9bb06
# ╠═c91862dd-498a-4712-8e3d-b77e088cd470
# ╠═a08d6e6d-ddc4-40aa-b7c4-93ea03191415
# ╟─a356e2cc-1cb1-457a-986c-998cf1efe008
# ╠═57141f7c-9261-4dc5-98e4-b136a15f86fc
# ╟─055e32d7-073c-40db-a267-750636b9f786
# ╠═aaa97ce4-a5ff-4332-89a2-843cee2e5b6d
# ╠═1067527e-76b7-4331-b3ab-efd72fb99dfc
# ╟─c4344e64-aa22-4328-a97a-71e44bcd289f
# ╟─827bda6f-87d4-4d36-8d89-f144f4595240
# ╠═0103b69a-2505-42f8-8df4-d08759eba077
# ╠═fda6171c-9675-4f2e-b226-7ccf100529cd
# ╠═d6564250-f646-40de-9463-a956af1a5b1d
# ╠═107aec28-ecb5-4007-95e5-25d0a7f0c465
# ╠═84e305e9-6ab9-4005-a295-a12c4eff68c5
# ╠═b8320f78-323c-49a9-a9f9-2748d19ecb35
# ╠═535716e6-9c1c-4324-a4cd-b1214df3c01d
# ╠═931a9c5f-8f91-4e88-956b-50c0efc9c58b
# ╠═7658a32c-d3da-4ec9-9d96-0d30bb18f08c
# ╟─e61981d5-5448-45e9-81dc-320ac87ba813
# ╠═4a75498d-8f4e-406f-8b01-f6a5f153919f
# ╠═31e1bb51-c531-4c4a-8634-5caafb7e9e51
# ╠═b0b81da4-6788-45c4-b618-188a02b5e09c
# ╠═8e618602-0c65-448f-adae-2c80e7cdd73e
# ╠═4cef9cea-1e84-42b9-bff6-b9a8b3bfe8da
# ╟─2871aca3-e6b4-4a2d-868a-36562e9a274c
# ╟─2a2e9155-1c77-46fd-8502-8431573f94d0
# ╠═7c792b6b-b6ee-4e30-88d5-d0b8064f2734
# ╠═febe8c06-b3aa-4db1-a3ea-fdc2a81bdebd
# ╠═a9981931-4cc9-4d16-a6d2-34b4071a84d7
# ╠═043d89a4-ac5d-49ac-9820-b35c5ee967bc
# ╟─b5008faf-fd43-45dd-a5a1-7f51e0b4ede5
# ╠═a756dd18-fac6-4527-944e-c16d8cc4bf95
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
