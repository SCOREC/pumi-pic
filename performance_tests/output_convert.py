# build-...-pumipic/performance_tests/output_convert.py

# Usage:    ./ps_combo $e $((e*1000)) $distribution -p $percent -n $struct
#           python output_convert.py
# Input: "filename"


# Formatting output from testing to comparison of rebuild/pseudopush
filename = input("Name of testing file: ").strip()
file = open(filename, "r")
lines = file.readlines()
file.close()

edit_lines = ["Sell-32-ne rebuild", "Sell-32-ne pseudo-push",
              "CSR rebuild", "CSR pseudo-push",
              "CabM rebuild", "CabM pseudo-push" ]

elms = 0

rebuild = open("rebuild_data.dat", "w")
push = open("push_data.dat", "w")
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