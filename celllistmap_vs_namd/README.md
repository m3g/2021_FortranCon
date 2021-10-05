# Comparision of CellListMap.jl with NAMD

## Important

The Julia simulation code in `simulate.jl` is not highly optimized (there is no parallelization of the updating of coordinates and velocities, for example). The aim here is to illustrate how effective is `CellListMap.jl` in building the cell lists to avoid unnecessary pairwise interactions. No electrostatic interactions are present in the system. `NAMD` is a full features simulation package, which does many things besides just computing short-ranged Lennard-Jones interactions. The input parameters were set to reduce at minimum the extra work it might be performing (but it is hard to tell which is the overhead of having the *possibility* of performing much more complex calculations), and also to force the package to recompute the pair lists at every step, which is not the standard procedure in a MD simulation. 

The only meaninful comparison here concerns the effectiveness of the `CellListMap.jl` cell list implementation for computing short-range interactions or other pairwise-dependent properties, which is good enough for the performance of this computation be comparable with that of a package as developed as NAMD.

## Performance in a simulation of 10k Neon atoms

These benchmarks were run on a Samsung i7 8th gen laptop, using 8 threads.

## Run the benchmark

After installing NAMD, Julia, and the package dependencies (as explained below), execute the benchmark with:
```bash
./run.sh
```

The expected output is something like:
```
% ./run.sh 

NAMD:

real    1m18,081s
user    9m31,554s
sys     0m1,150s

CellListMap:

real    1m4,829s
user    6m28,151s
sys     0m1,669s
```

and two trajectory files will be created: `ne10k.dcd` by `NAMD`, and `ne10k_traj.xyz` by `simulate.jl`. These trajectories can be visualized with `vmd -e view_namd.vmd`  and `vmd -e view_julia.vmd`, and should look like some blue atoms shaking a little. 

## Code

The Julia simulation code is available in the `simulate.jl` file. To execute it `Julia 1.6+` is required, and some package need to be installed. After installing Julia, launch a Julia REPL with

```bash
julia -t 8
```
(where `8` the number of threads desired), and do:

```julia
julia> import Pkg

julia> packages = ["Chemfiles", "CellListMap", "FastPow", "StaticArrays", "Parameters" ]

julia> Pkg.add(packages)

julia> include("./simulate.jl")

```

The NAMD program can be downloaded and installed from [the NAMD site](https://www.ks.uiuc.edu/Research/namd/).
