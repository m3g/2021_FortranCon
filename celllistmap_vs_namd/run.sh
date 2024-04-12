echo "
NAMD:"
time /home/leandro/programs/NAMD_3.0b6_Linux-x86_64-multicore/namd3 +p8 ne10k.namd >& ne10k.log

echo "
CellListMap:"
time julia --project -t 8 simulate.jl >& simulate.log 
