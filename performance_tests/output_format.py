# build-...-pumipic/performance_tests/output_format.py

# Formatting output from testing to just the numbers

print('Input number of tests:')
num_tests = int(input())

print('Input number of structures per test:')
num_structures = int(input())


with open('out.txt') as f:
    with open('formatted_out.txt',mode='w') as o:
        #num_tests,num_structures
        o.write(str(num_tests) +' ' + str(num_structures)+ '\n') 
        for i in range(num_tests):
            #get through MPI errors
            for j in range(23):
                f.readline() 
            #Test command info
            f.readline()
            o.write(f.readline())

            for j in range(num_structures):
                f.readline()
            f.readline() #Timing summary
            
            for j in range(num_structures):
                l = f.readline().split(' ')
                x = l[-1].split('=')
                o.write(l[0][:-1] + ','+x[1])
            f.readline()

with open('formatted_out.txt') as f:
    with open('finished_data.txt',mode='w') as o:
        o.write('ne,np,dist,% moved,CSR (time),SCS (time), SCS/CSR\n')
        x = f.readline().split()
        num_tests = int(x[0])
        num_structures = int(x[1])
        for i in range(num_tests):
            s = f.readline().split(' ')
            print(s)
            ne = s[2]
            np = s[3]
            dist = s[4]
            percent = s[5][:-1]
            average = 0.0
            for j in range(num_structures - 1):
                average += float(f.readline().split(',')[1][:-2])
            average /= num_structures-1
            print('SCS\t\t\tCSR')
            csr_time = f.readline().split(',')[1][:-1]
            print(str(average) + '\t' + csr_time+'\n')
            o.write(ne+','+np+','+dist+','+percent+','+csr_time+','+str(average)+','+str(average/float(csr_time))+'\n')
            
