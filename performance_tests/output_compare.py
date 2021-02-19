# build-...-pumipic/performance_tests/output_compare.py

# Usage: python output_compare.py
# Input: "filename"

# Formatting output from testing for usage in MATLAB
filename = input("Name of testing file: ").strip()
file = open(filename, "r")
lines = file.readlines()
file.close()

edit_lines = ["Sell-32-ne rebuild", "Sell-32-ne pseudo-push",
              "CSR rebuild", "CSR pseudo-push",
              "CabM rebuild", "CabM pseudo-push" ]

rebuild = []
push = []

print_command = True

output = open("rebuild_pseudo_comp.txt", "w")
for line in lines:
    # command line
    if print_command:
        if "./" in line:
            line = line.strip()
            line = line.split(" -n")[0] + "\n"
            output.write(line)
            output.write("Pseudo-Push Averages:\n")
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

            # output
            if "Sell-32-ne" in name:
                print_command = False
            elif "CabM rebuild" in name:
                for test in rebuild:
                    output.write(test)
                rebuild = []
                output.write("\n")
                print_command = True
            elif "CabM pseudo-push" in name:
                for test in push:
                    output.write(test)
                push = []
                output.write("Rebuild Averages:\n")

output.close()