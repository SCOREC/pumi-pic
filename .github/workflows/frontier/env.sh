module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=1