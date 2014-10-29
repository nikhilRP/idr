# Build steps
file(GLOB_RECURSE IDR_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.C)

add_library(${MYLIB_NAME} STATIC ${IDR_SRC_FILES})
add_library(${MYLIB_NAME}Shared SHARED ${IDR_SRC_FILES})
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(../idr ${IDR_SRC_FILES})
