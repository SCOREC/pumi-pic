# Formatting for ps_combo.cpp output when sent to file with &> # Note: only works on output created by compute nodes

import sys
#input arguments:
# argv[1] = num tests
# argv[2] = num structures

if len(sys.argv) != 3:
    print('Incorrect arguments')
    print('Usage: python3 ' + sys.argv[0] + ' <num tests> <num structures>')
    exit()

tests = int(sys.argv[1])
structures = int(sys.argv[2])

with open('out.txt') as f:
    with open('new_optimal_comparison.txt', mode='a') as o:
        for k in range(tests):
            for i in range(24):
                f.readline()
            # Test command info
            o.write(f.readline())
            # Gen
            f.readline()
            # Structure build info
            for i in range(structures):
                o.write(f.readline())
            # Get to timing
            for i in range(2+2*structures):
                f.readline()
            # Timing header
            o.write(f.readline())

            for i in range(structures):
                o.write(f.readline()) # push
                for j in range(5):
                    s = f.readline()
                    if s.split(' ')[0] == 'redistribute':
                        f.readline()
                o.write(f.readline()) # rebuild
            f.readline() #blank line at end of output



###############################################################################
# EXAMPLE of what is being formatted                                          #
###############################################################################
"""
[dcs211.ccni.rpi.edu:118501] mca_base_component_repository_open: unable to open mca_plm_lsf: libbat.so: cannot open shared object file: No such file or directory (ignored)
[dcs211.ccni.rpi.edu:118501] mca_base_component_repository_open: unable to open mca_ras_lsf: libbat.so: cannot open shared object file: No such file or directory (ignored)
--------------------------------------------------------------------------
By default, for Open MPI 4.0 and later, infiniband ports on a device
are not used by default.  The intent is to use UCX for these devices.
You can override this policy by setting the btl_openib_allow_ib MCA parameter
to true.

Local host:              dcs211
Local adapter:           mlx5_1
Local port:              1

--------------------------------------------------------------------------
--------------------------------------------------------------------------
WARNING: There was an error initializing an OpenFabrics device.

Local host:   dcs211
Local device: mlx5_1
--------------------------------------------------------------------------
[dcs211.ccni.rpi.edu:118490] mca_base_component_repository_open: unable to open mca_pml_pami: libpami.so.3: cannot open shared object file: No such file or directory (ignored)
[dcs211.ccni.rpi.edu:118490] mca_base_component_repository_open: unable to open mca_coll_ibm: libcollectives.so.3: cannot open shared object file: No such file or directory (ignored)
[dcs211.ccni.rpi.edu:118490] mca_base_component_repository_open: unable to open mca_coll_hcoll: libsharp_coll.so.4: cannot open shared object file: No such file or directory (ignored)
[dcs211.ccni.rpi.edu:118490] mca_base_component_repository_open: unable to open mca_osc_pami: libpami.so.3: cannot open shared object file: No such file or directory (ignored)
Test Command:
 ./ps_combo 100 10000 3 -s 64 -v 512 -n 2
Generating particle distribution with strategy: Exponential
Building SCS with C: 6 sigma: 100 V: 512
Building CSR
Performing 100 iterations of rebuild on each structure
Beginning push on structure Sell-64-ne
Beginning rebuild on structure Sell-64-ne
Beginning push on structure CSR
Beginning rebuild on structure CSR
Timing Summary 0
Operation                           Total Time   Call Count   Average Time
Sell-64-ne push                       0.259141          100       0.002591
redistribute                          4.253045          200       0.021265
Sell-64-ne count active particles     0.033687          100       0.000337
Sell-64-ne shuffle attempt            0.060826          100       0.000608
Sell-64-ne SCS specific building      0.214875          100       0.002149
Sell-64-ne PSToPs                     0.146113          100       0.001461
Sell-64-ne ViewsToViews               0.005066          100       0.000051
Sell-64-ne rebuild                    0.467501          100       0.004675
CSR push                              0.039731          100       0.000397
CSR count active particles            0.004612          100       0.000046
CSR calc ppe                          0.020249          100       0.000202
CSR offsets and indices               0.048192          100       0.000482
CSR PSToPS                            0.073495          100       0.000735
CSR ViewsToViews                      0.005280          100       0.000053
CSR rebuild                           0.184015          100       0.001840
"""
# BECOMES
"""
 ./ps_combo 100 10000 3 -s 64 -v 512 -n 2
Building SCS with C: 6 sigma: 100 V: 512
Building CSR
Operation                           Total Time   Call Count   Average Time
Sell-64-ne push                       0.259141          100       0.002591
Sell-64-ne rebuild                    0.467501          100       0.004675
CSR push                              0.039731          100       0.000397
CSR rebuild                           0.184015          100       0.001840
"""
