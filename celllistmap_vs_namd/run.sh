echo "
NAMD:"
time ~/programs/NAMD_2.12_Linux-x86_64-multicore/namd2 +p8 ne10k.namd >& ne10k.log

echo "
CellListMap:"
time julia -t 8 simulate.jl >& simulate.log 
