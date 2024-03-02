# build-...-pumipic/performance_tests/output_convert.py

# Usage:    ./tests ... &> input_filename
#           OR
#           python output_convert.py input_filename rebuild_output_filename push_output_filename migrate_output_filename

import sys
import re

# Formatting output from testing for usage in MATLAB
n = len(sys.argv)
assert( n >= 4 )

exeName="ps_combo"
inputfile = sys.argv[1]
rebuildfile = sys.argv[2]
pushfile = sys.argv[3]
migratefile = None
if n >= 5: migratefile = sys.argv[4]

file = open(inputfile, "r")
lines = file.readlines()
file.close()

edit_lines = ["CSR rebuild", "CSR pseudo-push", "CSR particle migration",
              "CabM rebuild", "CabM pseudo-push", "CabM particle migration",
              "DPS rebuild", "DPS pseudo-push", "DPS particle migration" ]

scs_pattern = re.compile(r'Sell-[0-9]+-ne [\brebuild\b | \bpseudo-push\b | \bparticle migration\b]')

elms = 0

rebuild = open(rebuildfile, "w")
push = open(pushfile, "w")
migrate = None
if n >= 5: migrate = open(migratefile, "w")

# write header
structures = {0:"SCS",
              1:"CSR",
              2:"CabM",
              3:"DPS"}
distributions = {0:"Evenly",
                 1: "Uniform",
                 2: "Gaussian",
                 3: "Exponential",
                 4: "GITRm Approximation"}
stmt="{0} {1} {2} {3}\n".format("structure", "elements", "distribution", "average time (s)")
for output in [rebuild, push, migrate]:
    if output is not None:
        output.write("structures: {}\n".format(structures))
        output.write("distributions: {}\n".format(distributions))
        output.write(stmt)

for line in lines:
    # command line
    if exeName in line:
        tokens = line.strip().split()
        elm = tokens[1]
        distribution = tokens[3]
        structure = tokens[4]
    # timing

    foundStructure = (scs_pattern.match(line) is not None)
    if not foundStructure:
        for check in edit_lines:
            if check in line:
                foundStructure = True
    if foundStructure:
        line = line.split()
        function = line[1]
        if "particle" in function:
            function = function + line[2]
            average = line[5]
        else:
            average = line[4]

        if "rebuild" in function:
            rebuild.write( "{0} {1} {2} {3}\n".format(structure, elm, distribution, average) )
        elif "pseudo-push" in function:
            push.write( "{0} {1} {2} {3}\n".format(structure, elm, distribution, average) )
        elif "migration" in function and (n >= 5):
            migrate.write( "{0} {1} {2} {3}\n".format(structure, elm, distribution, average) )

rebuild.close()
push.close()
migrate.close()
