cmake_minimum_required(VERSION 3.20)


SET(CTEST_DO_SUBMIT ON)
SET(CTEST_TEST_TYPE Nightly)

set(CTEST_SITE             "cranium.scorec.rpi.edu" )
set(CTEST_DASHBOARD_ROOT   "/lore/cwsmith/nightlyBuilds/pumipic")
set(CTEST_CMAKE_GENERATOR  "Unix Makefiles" )
set(CTEST_BUILD_CONFIGURATION  RelWithDebInfo)

set(CTEST_PROJECT_NAME "pumipic")
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
set(CTEST_DROP_LOCATION "/submit.php?project=pumi-pic")
set(CTEST_DROP_SITE_CDASH TRUE)

find_program(CTEST_GIT_COMMAND NAMES git)
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

ctest_start(${CTEST_TEST_TYPE})


if(CTEST_DO_SUBMIT)
  ctest_submit(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)
  if(HAD_ERROR)
    message(FATAL_ERROR "Cannot submit pumi-pic Project.xml!")
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

  setup_repo(${subproject_name} "git@github.com:SCOREC/pumi-pic.git")

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

set(OMEGAH_1050_INSTALL
  "${CTEST_DASHBOARD_ROOT}/build-omegah1050-cranium-cuda114/install/lib/cmake/Omega_h")
SET(CONFIGURE_MASTER_OMEGAH1050
  "-DCMAKE_CXX_COMPILER=mpicxx"
  "-DIS_TESTING=ON"
  "-DPS_IS_TESTING=ON"
  "-DOmega_h_PREFIX=${OMEGAH_1050_INSTALL}"
  "-DTEST_DATA_DIR=${CTEST_DASHBOARD_ROOT}/repos/pumipic/pumipic-data")

set(OMEGAH_MASTER_INSTALL
  "${CTEST_DASHBOARD_ROOT}/build-omegah-cranium-cuda114/install/lib/cmake/Omega_h")
SET(CONFIGURE_MASTER_OMEGAH_MASTER
  "-DCMAKE_CXX_COMPILER=mpicxx"
  "-DIS_TESTING=ON"
  "-DPS_IS_TESTING=ON"
  "-DOmega_h_PREFIX=${OMEGAH_MASTER_INSTALL}"
  "-DTEST_DATA_DIR=${CTEST_DASHBOARD_ROOT}/repos/pumipic/pumipic-data")

message(STATUS "configure options ${CONFIGURE_MASTER_OMEGAH1050}")
build_subproject(pumipic-master-omegah1050 "${CONFIGURE_MASTER_OMEGAH1050}")
test_subproject(pumipic-master-omegah1050)

message(STATUS "configure options ${CONFIGURE_MASTER_OMEGAH_MASTER}")
build_subproject(pumipic-master-omegahMaster "${CONFIGURE_MASTER_OMEGAH_MASTER}")
test_subproject(pumipic-master-omegahMaster)

message("DONE")
