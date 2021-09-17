# Comparing the codes and performances of two simple 2D simulations

## Important

These codes are *not* optimized for ultimate performance. Optimizations can be performed on both sides. The present comparison is only to illustrate that propertly written Julia and Fortran codes should look and perform similarly. The syntax of both languages is very idiomatic for the representation of numerical calculations. 

## Run the benchmark

To execute the benchmark, just do:
```bash
./run.sh
```

It is expected that `Julia` and `gfortran` are installed, and that the `StaticArrays` package was installed in `Julia`.  

The expected output is something like:
```
Fortran:

real    0m49,351s
user    0m49,306s
sys     0m0,020s

Julia:

real    0m47,848s
user    0m47,064s
sys     0m0,693s
```

and two trajectory files will be created: `traj_fortran.xyz` and `traj_julia.xyz`, which can be visualized in VMD, for example. The trajectory should look like this: https://youtu.be/GSyDRJUYWYY. These trajectories can be visualized with `vmd -e view_fortran.vmd`  and `vmd -e view_julia.vmd`.

Small variations are expected, but the performances will be probably similar.

## Code comparison

Here the syntax of the two codes being run are put side by side, for comparison. The author's opinion is that both languages are very simple and clear. 

### Function `wrap`:

The wrap function computes the minimum-image distance associated to a coordinate. The codes are very similar in every aspect, except that the Fortran code is restricted to the type of floats defined in `dp`. The Fortran function was declared `elemental` to be used on arrays, element-wise. 

<table width=100%>
<tr><td align=center><b>Julia</b></td><td align=center><b>Fortran</b></td></tr>
<tr>
<td valign=top>

```julia
function wrap(x,side)
    x = mod(x,side)
    if x >= side/2
        x -= side
    elseif x < -side/2
        x += side
    end
    return x
end  
```

</td><td valign=top>

```fortran
elemental real(dp) function wrap(x,side)
    implicit none
    real(dp), intent(in) :: x, side
    wrap = dmod(x,side)
    if (wrap >= side/2) then
        wrap = wrap - side
    else if (wrap < -side/2) then
        wrap = wrap + side
    end if
end function wrap
```

</td></tr></table>

### Function `force_pair`:

This function computes the forces given the coordinates of two particles as the input. The differences here are that in the Julia code the particles are of a generic type `T`, which can be of any dimension and/or variable type. The Fortran code is restricted here to floats of type `dp`, and the dimension of the points is defined by the `ndim` input variable. Writing one explicit loop can be made shorter with the broadcasting (`.`) Julia syntax, which is used here for `wrap`, and will be seen in other places. The Fortran `wrap` function, by being declared `elemental`, is automatically broadcasted.  

<table width=100%>
<tr><td align=center><b>Julia</b></td><td align=center><b>Fortran</b></td></tr>
<tr>
<td valign=top>

```julia
function force_pair(x::T,y::T,cutoff,side) where T
    Δv = wrap.(y - x, side)
    d = norm(Δv)
    if d > cutoff
        return zero(T)
    else
        return (d - cutoff)*(Δv/d)
    end
end
```

</td><td valign=top>

```fortran
subroutine force_pair(ndim,fx,x,y,cutoff,side)
    implicit none
    integer :: ndim
    real(dp) :: fx(ndim), d
    real(dp) :: x(ndim), y(ndim), cutoff, side, dv(ndim)
    dv = wrap(y-x,side)
    d = norm(ndim,dv)
    if (d > cutoff) then
        fx = 0.0_dp
    else
        fx = (d - cutoff)*(dv/d)
    end if
end subroutine force_pair
```

</td></tr></table>

### Function `forces`:

The codes are very similar, with the exception that we have opted to use use `fill!` function in Julia which is the preferred syntax in this case because `f` is an array of arrays. The use of `zero(T)` allows the code to be generic. An `@inbounds` flag was used in this inner loop which is critical for performance, and results in a small, but noticeable performance gain (about 5%). The Fortran code is very compact as well, being lengthier only as a result of type declarations. 

<table width=100%>
<tr><td align=center><b>Julia</b></td><td align=center><b>Fortran</b></td></tr>
<tr>
<td valign=top>

```julia
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
```

</td><td valign=top>

```fortran
subroutine forces(n,ndim,f,x,cutoff,side)
    implicit none
    integer :: n, ndim, i, j
    real(dp) :: f(ndim,n), x(ndim,n)
    real(dp) :: fx(ndim)
    real(dp) :: cutoff, side
    f = 0.0_dp
    do i = 1, n-1
        do j = i+1, n
            call force_pair(ndim,fx,x(:,i),x(:,j),cutoff,side)
            f(:,i) = f(:,i) + fx
            f(:,j) = f(:,j) - fx
        end do
    end do
end subroutine forces
```

</td></tr></table>

### Function `md`:

Again the codes are very similar. The modern vectorized Fortran syntax allows the updating of the accelerations, positions and velocities in a simple and compact as well. We use the `@.`  syntax in Julia to indicate operations over all elements of the vectors involved. 

In Julia we create an array of arrays to store the saved trajectory, and new steps will be pushed to this array. In Fortran we preallocate the complete array (I am not aware of something equivalent to `push!` in Fortran).

<table width=100%>
<tr><td align=center><b>Julia</b></td><td align=center><b>Fortran</b></td></tr>
<tr>
<td valign=top>

```julia
function md(
    x0,v0, mass,dt,
    nsteps,isave, force_pair::F
) where F
    x = copy(x0)
    v = copy(v0)
    f = similar(x0)
    a = similar(x0)
    # Save initial point
    trajectory = [ copy(x0) ]
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
```

</td><td valign=top>

```fortran
subroutine md(n,ndim,x0,v0,mass,dt,nsteps,isave,trajectory,cutoff,side)
    implicit none
    integer :: n, ndim, i, step, nsteps, isave, isaved
    real(dp) :: dt
    real(dp) :: x0(ndim,n), v0(ndim,n), mass(n)
    real(dp) :: x(ndim,n), v(ndim,n), f(ndim,n), a(ndim,n)
    real(dp) :: trajectory(ndim,n,nsteps/isave+1)
    real(dp) :: cutoff, side
    ! Save initial positions
    trajectory(:,:,1) = x0
    x = x0
    v = v0
    isaved = 1
    do step = 1, nsteps
        ! Compute forces
        call forces(n,ndim,f,x,cutoff,side)
        ! Update positions and velocities 
        do i = 1, n
           a(:,i) = f(:,i) / mass(i)
           x(:,i) = x(:,i) + v(:,i)*dt + a(:,i)*dt**2/2
           v(:,i) = v(:,i) + a(:,i)*dt
        end do
        ! Save if required
        if (mod(step,isave) == 0) then
            isaved = isaved + 1
            trajectory(:,:,isaved) = x
        end if
    end do
end subroutine md
```

</td></tr></table>


### Main program

Here the codes differ a little bit, because the generation of initial coordinates and velocities can be performed with comprehensions `[...]` in Julia, and in Fortran we use explicit loops over preallocated arrays. The difference in the data structure becomes apparent, as we use vectors of `Point2D` objects in Julia, and plain matrices in Fortran. The memory layout of these two structures is the same, but abstracting the type of object as done in Julia allows the code to be more generic in general. The `md` function in Julia is called with the expansion of a named tuple for additional clarity.  

In Julia the use of the module variables is declared outside the function, while in Fortran the module is used inside the main program. Finally, the Julia ends with a call to the `main()` function.

<table width=100%>
<tr><td align=center><b>Julia</b></td><td align=center><b>Fortran</b></td></tr>
<tr>
<td valign=top>

```julia
using .ParticleSimulation

function main()
    n = 100
    cutoff = 5.
    side = 100.
    nsteps = 200_000
    trajectory = md((
        x0 = [random_point(Point2D{Float64},(-50,50)) for _ in 1:n ], 
        v0 = [random_point(Point2D{Float64},(-1,1)) for _ in 1:n ], 
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
```

</td><td valign=top>

```fortran
program main
    use ParticleSimulation
    implicit none
    integer, parameter :: n = 100, ndim = 2
    integer :: i, j, k, nsteps, isave
    real(dp) :: x0(ndim,n), v0(ndim,n), mass(n)
    real(dp) :: dt
    real(dp) :: cutoff, side
    real(dp), allocatable :: trajectory(:,:,:)
    ! Initialize positions and velocities
    do i = 1, n
        do j = 1, ndim
            x0(j,i) = -50 + 100*dp_rand()
            v0(j,i) = -1 + 2*dp_rand()
        end do
        mass(i) = 10
    end do
    ! Parameters
    dt = 0.001_dp
    nsteps = 200000
    isave = 1000
    cutoff = 5.0_dp
    side = 100.0_dp
    allocate(trajectory(ndim,n,nsteps/isave + 1))
    ! Run simulation
    call md(n,ndim,x0,v0,mass,dt,nsteps,isave,trajectory,cutoff,side)
    open(10,file="traj_fortran.xyz")
    k = 0
    do i = 1, nsteps/isave + 1
        k = k + 1
        write(10,*) n
        write(10,*) " step = ", k
        do j = 1, n
            write(10,*) "He", wrap(trajectory(1,j,k),side), wrap(trajectory(2,j,k),side), 0.0_dp
        end do
    end do
    close(10)
end program main
```

</td></tr></table>

### Additional code

In the Julia implementation we need to define the `Point2D` data structure, which is set as a subtype of the convenient `FieldVector` structure of the `StaticArrays` package, which allow all arithmetics to work out of the box for this type of point. We use also the `Printf` package to write the coordinates, and import the `norm`  function from `LinearAlgebra` (which is available by default in Julia), although writing a custom `norm` function would be trivial and equivalent. We also defined a custom function to generate random points, and the code ends with the explicit call to the main function with the desired number of steps. 

On the Fortran side we define the type of floats of the package explicitly. A simple function to return a random number was defined and a function to compute the norm of a vector of the desired dimension was required.

<table width=100%>
<tr><td align=center><b>Julia</b></td><td align=center><b>Fortran</b></td></tr>
<tr>
<td valign=top>

```julia
using StaticArrays
using Printf
using LinearAlgebra: norm

struct Point2D{T} <: FieldVector{2,T}
    x::T
    y::T
end

function random_point(::Type{Point2D{T}},range) where T 
    p = Point2D(
        range[begin] + rand(T)*(range[end]-range[begin]),
        range[begin] + rand(T)*(range[end]-range[begin])
    )
    return p
end
```

</td><td valign=top>

```fortran
integer, parameter :: dp = kind(0.0d0)

double precision function dp_rand()
    call random_number(dp_rand)
end function dp_rand

double precision function norm(ndim,x)
    integer :: ndim
    double precision :: x(ndim)
    norm = 0.0_dp
    do i = 1, ndim
        norm = norm + x(i)**2
    end do
    norm  = dsqrt(norm)
end function norm
```

</td></tr></table>

