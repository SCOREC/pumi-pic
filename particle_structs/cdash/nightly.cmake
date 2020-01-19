cmake_minimum_required(VERSION 2.8)


SET(CTEST_DO_SUBMIT ON)
SET(CTEST_TEST_TYPE Nightly)

set(CTEST_SITE             "blockade.scorec.rpi.edu" )
set(CTEST_DASHBOARD_ROOT   "/fasttmp/cwsmith/nightlyBuilds" )
set(CTEST_CMAKE_GENERATOR  "Unix Makefiles" )
set(CTEST_BUILD_CONFIGURATION  RelWithDebInfo)

set(CTEST_PROJECT_NAME "particle_structures")
set(CTEST_SOURCE_NAME repos)
set(CTEST_BUILD_NAME  "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set(CTEST_BINARY_NAME build_${CTEST_PROJECT_NAME})

set(CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set(CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file(MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif()
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif()

configure_file(${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
               ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")
set(CTEST_BUILD_FLAGS -j8)

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=particle_structures")
set(CTEST_DROP_SITE_CDASH TRUE)

find_program(CTEST_GIT_COMMAND NAMES git)
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

ctest_start(${CTEST_TEST_TYPE})


if(CTEST_DO_SUBMIT)
  ctest_submit(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)
  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot submit particle_structures Project.xml!")
  endif()
endif()


macro(submit_part subproject_name part)
  if(CTEST_DO_SUBMIT)
    ctest_submit(PARTS ${part} RETURN_VALUE HAD_ERROR)
    if(HAD_ERROR)
      message(FATAL_ERROR "Cannot submit ${subproject_name} ${part} results!")
    endif()
  endif()
endmacro()

macro(build_subproject subproject_name config_opts)
  set_property(GLOBAL PROPERTY SubProject ${subproject_name})
  set_property(GLOBAL PROPERTY Label ${subproject_name})

  setup_repo(${subproject_name} "git@github.com:SCOREC/particle_structures.git")

  if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/${subproject_name}")
    file(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/${subproject_name})
  endif()

  ctest_configure(
    BUILD "${CTEST_BINARY_DIRECTORY}/${subproject_name}" 
    SOURCE "${CTEST_SOURCE_DIRECTORY}/${subproject_name}"
    OPTIONS "${config_opts}"
    RETURN_VALUE HAD_ERROR)

  submit_part(${subproject_name} Configure)

  ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}/${subproject_name}"
              RETURN_VALUE HAD_ERROR)

  submit_part(${subproject_name} Build)
endmacro()

macro(test_subproject subproject_name)
  ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}/${subproject_name}")
  submit_part(${subproject_name} Test)
endmacro()

macro(setup_repo repo_name repo_url)
  if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/${repo_name}")
    EXECUTE_PROCESS(COMMAND "${CTEST_GIT_COMMAND}" 
                    clone ${repo_url} ${CTEST_SOURCE_DIRECTORY}/${repo_name}
                    RESULT_VARIABLE HAD_ERROR)
    if(HAD_ERROR)
      message(FATAL_ERROR "Cannot checkout ${repo_name} repository!")
    endif()
  endif()

  ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/${repo_name}" RETURN_VALUE count)
  message("Found ${count} changed files")
  submit_part(${repo_name} "Update")
endmacro(setup_repo)

SET(CONFIGURE_MASTER "-DENABLE_KOKKOS=ON")

message(STATUS "configure options ${CONFIGURE_MASTER}")
build_subproject(master "${CONFIGURE_MASTER}")
test_subproject(master)

message("DONE")
