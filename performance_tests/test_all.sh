#!/bin/bash

cd ~/barn/pumipic_CabM/pumi-pic/performance_tests/

./test_smallE_largeP.sh &> smallE_largeP_$SLURM_JOB_ID.txt
cd ~/barn/pumipic_CabM/pumi-pic/performance_tests/
python output_convert.py smallE_largeP_$SLURM_JOB_ID.txt smallE_largeP_rebuild.dat smallE_largeP_push.dat smallE_largeP_migrate.dat
echo "test_smallE_largeP DONE"

./test_largeE_smallP.sh &> largeE_smallP_$SLURM_JOB_ID.txt
cd ~/barn/pumipic_CabM/pumi-pic/performance_tests/
python output_convert.py largeE_smallP_$SLURM_JOB_ID.txt largeE_smallP_rebuild.dat largeE_smallP_push.dat largeE_smallP_migrate.dat
echo "test_largeE_smallP DONE"