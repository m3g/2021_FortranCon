gfortran -O3 -march=native particle_simulation.f90 -o particle_simulation

echo "Fortran:"
time ./particle_simulation

echo "Julia:"
time julia particle_simulation.jl
