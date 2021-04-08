# build-...-pumipic/performance_tests/output_compare.py

# Usage:    ./tests &> input_filename
#           OR
#           python output_convert.py input_filename output_filename

import sys

# Formatting output from testing to comparison of rebuild/pseudopush/migration
n = len(sys.argv)

inputfile = sys.argv[1]
outputfile = sys.argv[2]

input_stream = open(inputfile, "r")
lines = input_stream.readlines()
input_stream.close()

edit_lines = ["Sell-32-ne rebuild", "Sell-32-ne pseudo-push", "Sell-32-ne particle migration",
              "CSR rebuild", "CSR pseudo-push", "CSR particle migration",
              "CabM rebuild", "CabM pseudo-push", "CabM particle migration" ]

rebuild = []
push = []
migrate = []

print_command = True

output = open(outputfile, "w")
for line in lines:
    # command line
    if print_command:
        if "./" in line:
            line = line.strip()
            line = line.split(" -n")[0] + "\n"
            output.write(line)
    # timing
    for check in edit_lines:
        if check in line:
            line = line.split()
            name = line[0] + " " + line[1]
            average = line[4]
            
            # add to lists to get in order
            if "rebuild" in name:
                rebuild.append( "     {0:<30} {1}\n".format(name,average) )
            elif "pseudo-push" in name:
                push.append( "     {0:<30} {1}\n".format(name,average) )
            elif "particle" in name and "migration" in line[2]:
                name = name + " " + line[2]
                average = line[5]
                migrate.append( "     {0:<30} {1}\n".format(name,average) )

            # output
            if "Sell" in name:
                print_command = False
            elif "CabM particle migration" in name:
                output.write("Migrate Averages:\n")
                for test in migrate:
                    output.write(test)
                migrate = []
                print_command = True
            elif "CabM pseudo-push" in name:
                output.write("Pseudo-Push Averages:\n")
                for test in push:
                    output.write(test)
                push = []
            elif "CabM rebuild" in name:
                output.write("Rebuild Averages:\n")
                for test in rebuild:
                    output.write(test)
                rebuild = []
                output.write("\n")

output.close()