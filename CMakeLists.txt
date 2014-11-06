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

# default to release builds
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(RELEASE_FLAGS "-O4 -DNDEBUG")
set(DEBUG_FLAGS "-g -O0 -Wall")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_C_FLAGS ${RELEASE_FLAGS})

set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_C_FLAGS_DEBUG ${DEBUG_FLAGS})

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
