#ifndef PSPARAMS_H
#define PSPARAMS_H
const double TERA = 1E12;

//six operations = add and multiply each coordinate in 3D
//CUDA implements fused multiply add (FMA)[1], I'm counting that as two
// operations.
//[1] https://docs.nvidia.com/cuda/floating-point/index.html
const double PARTICLE_OPS = 6; 
#endif
