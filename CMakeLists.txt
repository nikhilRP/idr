cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(idr)

if($ENV{BUILD_NUMBER})
  set(BUILD_NUMBER $ENV{BUILD_NUMBER})
else()
  set(BUILD_NUMBER 0)
endif()

set(MYLIB_NAME "IDR")
set(MYLIB_VERSION "0.1.0.${BUILD_NUMBER}")
set(MYLIB_BRIEF "${MYLIB_NAME} is a C++ implementation of Irreproducible Discovery Rate")

# Not using this for now
#option(MYLIB_TESTS_ENABLED "Enable tests build" ON)

# code coverage options
set(GCOV_FLAGS "-fprofile-arcs -ftest-coverage")

set(DEBUG_FLAGS "-g -O0 -Wall")
set(RELEASE_FLAGS "-O3")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_C_FLAGS ${RELEASE_FLAGS})

set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${GCOV_FLAGS})
set(CMAKE_C_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_C_FLAGS_DEBUG ${GCOV_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS_DEBUG ${GCOV_FLAGS})

# TODO OP check how this works on multiple configurations generators
# available configurations in multiple configurations enabled generators
set(CMAKE_CONFIGURATION_TYPES Debug Release)
#message("CMAKE_CFG_INTDIR = ${CMAKE_CFG_INTDIR}")

add_subdirectory(src)

#if(MYLIB_TESTS_ENABLED)
#  add_subdirectory(test)
#endif()

# used by doc home and in packaging
set(README_FILE ${PROJECT_SOURCE_DIR}/README.md)

# Licensing and stuff like that
set(LICENSE_FILE ${PROJECT_SOURCE_DIR}/LICENSE-MIT)
install(FILES
    ${README_FILE}
    ${LICENSE_FILE}
  DESTINATION .
)
