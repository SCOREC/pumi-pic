# build-...-pumipic/performance_tests/output_convert.py

# Usage:    ./ps_combo $e $((e*1000)) $distribution -p $percent -n $struct
#           python output_convert.py input_filename rebuild_output_filename push_output_filename

import sys

# Formatting output from testing to comparison of rebuild/pseudopush
n = len(sys.argv)
assert( n == 4 )

inputfile = sys.argv[1]
rebuildfile = sys.argv[2]
pushfile = sys.argv[3]

file = open(inputfile, "r")
lines = file.readlines()
file.close()

edit_lines = ["Sell-32-ne rebuild", "Sell-32-ne pseudo-push",
            "CSR rebuild", "CSR pseudo-push",
            "CabM rebuild", "CabM pseudo-push" ]

elms = 0

rebuild = open(rebuildfile, "w")
push = open(pushfile, "w")
for line in lines:
    # command line
    if "./" in line:
        line = line.strip().split()
        elm = line[1]
        distribution = line[3]
        percentage = line[5]
        structure = line[7]
    # timing
    for check in edit_lines:
        if check in line:
            line = line.split()
            function = line[1]
            average = line[4]

            if "rebuild" in function:
                rebuild.write( "{0} {1} {2} {3} {4}\n".format(structure, elm, distribution, percentage, average) )
            elif "pseudo-push" in function:
                push.write( "{0} {1} {2} {3} {4}\n".format(structure, elm, distribution, percentage, average) )

rebuild.close()
push.close()
