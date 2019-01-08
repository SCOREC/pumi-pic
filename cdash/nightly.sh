#!/bin/bash -x

#load system modules
source /etc/profile.d/modules.sh
source /etc/profile


module load gcc/7.3.0-bt47fwr mpich/3.2.1-niuhmad cmake/3.13.1-ovasnmm omega-h/9.19.1-sxkanjb trilinos/develop-mejumh2

d=/fasttmp/cwsmith/nightlyBuilds
cd $d/repos/gitrm
git pull
cd $d
git submodule init
git submodule update
#remove old compilation
[ -d build_gitrm ] && rm -rf build_gitrm/

#run nightly test script
ctest -VV -D Nightly -S $d/repos/gitrm/cdash/nightly.cmake
