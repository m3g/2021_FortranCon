import Chemfiles
using CellListMap
using FastPow
using StaticArrays
using Printf
using Base.Threads
using Parameters
using Statistics: mean
using LinearAlgebra: norm_sqr

#
# Simulation setup
#
@with_kw struct Params{V,N,T,M,UnitCellType}
    x0::V = getcoor("./ne10k_initial.pdb")  
    temperature::T = 300.
    nsteps::Int = 2_000
    dt::T = 1.0 # fs
    ibath::Int = 10
    print_energy::Int = 50 
    print_traj::Int = 100
    trajfile::String = "ne10k_traj.xyz"
    cutoff::T = 12.
    box::Box{UnitCellType,N,T,M} = Box([ 46.37, 46.37, 46.37 ], cutoff, lcell=2)
    # Parameters for Neon
    mass::T = 20.17900 # g/mol 
    ε::T = 0.0441795 # kcal/mol
    σ::T = 2*1.64009 # Å
    kB::T = 0.001985875 # Boltzmann constant kcal / mol K
end

function potential_energy(d2,ε,σ,u)
    @fastpow u += ε*( σ^12/d2^6 - 2*σ^6/d2^3 )
    return u
end

function forces(x,y,i,j,d2,ε,σ,f)
    r = y - x
    @fastpow dudr = -12*ε*(σ^12/d2^7 - σ^6/d2^4)*r
    @inbounds f[i] = f[i] + dudr
    @inbounds f[j] = f[j] - dudr
    return f
end

# Kinetic energy and temperature 
compute_kinetic(v::AbstractVector,m) = (m/2)*sum(x -> norm_sqr(x), v)
compute_temp(kinetic,kB,n) = 2*kinetic/(3*kB*n)
compute_temp(v::AbstractVector,m,kB) = 2*compute_kinetic(v,m)/(3*kB*length(v))

# Remove drift from velocities
function remove_drift!(v)
    vmean = mean(v)
    v .= v .- Ref(vmean)
end

# Function to print output data
function print_data(istep,x,params,cl,kinetic,trajfile;parallel=parallel)
    @unpack print_energy, print_traj, kB, box, ε, σ = params
    if istep%print_energy == 0
        u = map_pairwise!( 
            (x,y,i,j,d2,output) -> potential_energy(d2,ε,σ,output),
            0., box, cl, parallel=parallel,
        ) 
        temp = compute_temp(kinetic,kB,length(x))
        @printf(
            "STEP = %8i U = %12.5f K = %12.5f TOT = %12.5f TEMP = %12.5f\n", 
            istep, u, kinetic, u+kinetic, temp
        )
    end
    if istep%print_traj == 0 && istep > 0
        println(trajfile,length(x))
        println(trajfile," step = ", istep)
        for i in 1:length(x)
           @printf(trajfile,"Ne %12.5f %12.5f %12.5f\n", ntuple(j -> x[i][j], 3)...)
        end
    end
    return nothing
end

# Read coordinates from NAMD-DCD file
function getcoor(file)
    traj = redirect_stdout(() -> Chemfiles.Trajectory(file), devnull)
    frame = Chemfiles.read_step(traj,0)
    Chemfiles.close(traj)
    return copy(reinterpret(reshape,SVector{3,Float64},Chemfiles.positions(frame)))
end

#
# Simulation
#
function simulate(params::Params{V,N,T,UnitCellType}; parallel=true) where {V,N,T,UnitCellType}
    @unpack x0, temperature, nsteps, box, dt, ε, σ, mass, kB = params
    trajfile = open(params.trajfile,"w")

    # To use coordinates in Angstroms, dt must be in 10ps. Usually packages
    # use ps and nm internally (thus multiply coordinates by 10 and divide
    # the timestep given in fs by 1000)
    dt = dt/100

    # Initial arrays
    x = copy(x0)
    f = similar(x)
    flast = similar(x)

    # Initial velocities
    v = randn(eltype(x),size(x))
    remove_drift!(v)
    # Adjust average to desidred temperature
    t0 = compute_temp(v,mass,kB) 
    @. v = v * sqrt(temperature/t0)

    # Build cell lists for the first time
    cl = CellList(x,box,parallel=parallel)

    # preallocate threaded output, since it contains the forces vector
    f .= Ref(zeros(eltype(f)))
    f_threaded = [ deepcopy(f) for _ in 1:nthreads() ]
    aux = CellListMap.AuxThreaded(cl)

    # Print data at initial point
    kinetic = compute_kinetic(v,mass)
    print_data(0,x,params,cl,kinetic,trajfile,parallel=parallel)

    # Simulate
    for istep in 1:nsteps

        # Update positions (velocity-verlet)
        @. x = x + v*dt + 0.5*(f/mass)*dt^2

        # Update cell lists
        cl = UpdateCellList!(x,box,cl,aux,parallel=parallel)

        # Reset forces
        flast .= f
        f .= Ref(zeros(eltype(f)))
        @threads for it in 1:nthreads()
            f_threaded[it] .= Ref(zeros(eltype(f)))
        end

        # Update forces
        map_pairwise!( 
            (x,y,i,j,d2,output) -> forces(x,y,i,j,d2,ε,σ,output),
            f, box, cl, parallel=parallel,
            output_threaded=f_threaded
        ) 
         
        # Update velocities
        @. v = v + 0.5*((flast + f)/mass)*dt 

        # Print data and output file
        kinetic = compute_kinetic(v,mass)
        print_data(istep,x,params,cl,kinetic,trajfile,parallel=parallel)

        # Isokinetic bath
        if istep%params.ibath == 0
            remove_drift!(v)
            temp = compute_temp(kinetic,kB,length(v))
            @. v = v * sqrt(temperature/temp)
        end

   end
   close(trajfile)

end

params = Params()
@time simulate(params)





