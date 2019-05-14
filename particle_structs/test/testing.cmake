add_test(NAME type_test COMMAND ./typeTest)

add_test(NAME rebuild COMMAND ./rebuild)

add_test(NAME lambdaTest COMMAND ./lambdaTest)

add_test(NAME migrateNothing COMMAND ./migrateTest)

add_test(NAME migrate4 COMMAND mpirun -np 4 ./migrateTest)
