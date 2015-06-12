cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("Dali")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# REQUIRED PACKAGES
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Gflags REQUIRED)
find_package(HiRedis REQUIRED)

# OPTIONAL PACKAGES
find_package(GooglePerfTools)
find_package(BLAS)
find_package(MKL)
find_package(GTest)
find_package(CUDA)

# CUDA BEGINS HERE
if (DEFINED WITH_CUDA)
    message("WITH_CUDA ${WITH_CUDA}")
else()
    if (CUDA_FOUND STREQUAL TRUE)
        SET(WITH_CUDA TRUE)
    else()
        SET(WITH_CUDA FALSE)
        message("CUDA not found - did you know that Dali has GPU support?")
    endif()
endif(DEFINED WITH_CUDA)


if (WITH_CUDA)
    add_definitions(-DDALI_USE_CUDA)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})
    LIST(APPEND CUDA_NVCC_FLAGS --compiler-options ${CUDA_EXTRA_FLAGS} -lineinfo -Xptxas -dlcm=cg  -use_fast_math -std=c++11)
    SET(CUDA_EXTRA_FLAGS "-fno-strict-aliasing")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda/lib")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_EXTRA_FLAGS}")
    add_definitions(-DMSHADOW_USE_CUDA=1)
else (WITH_CUDA)
    add_definitions(-DMSHADOW_USE_CUDA=0)
endif (WITH_CUDA)
# CUDA ENDS HERE



if (NOT HIREDIS_FOUND)
    message( FATAL_ERROR "ERROR: required dependency HiRedis not found.\n    > brew install hiredis")
endif(NOT HIREDIS_FOUND)

if (NOT GPERFTOOLS_FOUND)
    message("WARNING: consider installing gperftools for better multithreaded performance.\n    > brew install gperftools")
endif(NOT GPERFTOOLS_FOUND)

# This monstrosity came to live to alleviate two conditions:
#      1) ctest ignores colored output
#      2) make test does not build tests (wtf)
set(TESTS_MAIN "${PROJECT_SOURCE_DIR}/dali/tests_main.cpp")
function(add_testsuite TEST_TARGET TEST_DEPENDENCIES TEST_FILES)
    list(APPEND TEST_FILES ${TESTS_MAIN})
    if(GTEST_FOUND)
        add_executable(${TEST_TARGET} ${TEST_FILES})
        add_dependencies(${TEST_TARGET} ${TEST_DEPENDENCIES})
        target_link_libraries(${TEST_TARGET} ${TEST_DEPENDENCIES} ${GTEST_BOTH_LIBRARIES})
        add_test(${TEST_TARGET} ${TEST_TARGET})
        add_dependencies(build_tests ${TEST_TARGET})
        add_custom_target(${TEST_TARGET}_run ${TEST_TARGET})
        add_dependencies(run_tests ${TEST_TARGET}_run)
    endif(GTEST_FOUND)
endfunction(add_testsuite)

if(GTEST_FOUND)
    enable_testing()
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_custom_target(run_tests true)
    add_custom_target(build_tests)
    add_dependencies(run_tests build_tests)
    ADD_CUSTOM_TARGET(run_tests_heapcheck
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/test_runner.sh ${CMAKE_BUILD_DIR}
        WORKING_DIRECTORY ${CMAKE_BUILD_DIR})
else(GTEST_FOUND)
    message("WARNING: GTESTS not found - install gtests to be able to run tests")
endif(GTEST_FOUND)

if (GPERFTOOLS_FOUND)
    add_definitions(-DGPERFTOOLS_FOUND)
endif(GPERFTOOLS_FOUND)

# CLEANING UP AFTER FINDERS
# some library list variables are FALSE by default if libs are not found,
# but it is much more useful to keep it empty.
set(LIBVARS_TO_FIX MKL_LIBRARIES BLAS_LIBRARIES GPERFTOOLS_TCMALLOC_AND_PROFILER)
foreach(libvar ${LIBVARS_TO_FIX})
    if (NOT ${libvar})
        set(${libvar} "")
    endif (NOT ${libvar})
endforeach(libvar)

# add submodule libraries
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/redox)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/json11)

# Relative path in C++ are hard. Adding a macro:
add_definitions(-DDALI_DATA_DIR=${PROJECT_SOURCE_DIR}/data)

# GFLAGS STUFF
add_definitions(-DGFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

# COMPILER SETUP
add_definitions(-DBUILD_DIR=${CMAKE_BUILD_DIR})

message("=> BUILD_TYPE " ${CMAKE_BUILD_TYPE})
if (APPLE)
    set (CMAKE_CXX_COMPILER "clang++")
#else (APPLE)
    # land of Linux or Windows
endif(APPLE)


if(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")
else(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w -fPIC")
endif(CMAKE_BUILD_TYPE MATCHES "debug")

# mshadow
add_definitions(-DMSHADOW_USE_CBLAS)
add_definitions(-DMSHADOW_USE_MKL=0)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/mshadow)

# INCLUDES
include_directories(${MKL_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${HIREDIS_INCLUDE_DIRS})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/redox/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/json11)

if (GPERFTOOLS_FOUND)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
endif (GPERFTOOLS_FOUND)

# ADDING DIRECTORIES WITH THEIR OWN CMAKELISTS FILES
add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf)
add_subdirectory(${PROJECT_SOURCE_DIR}/dali)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)


# for special SQLiteCpp target:
#file(GLOB_RECURSE SQLITECPP_SRC "${PROJECT_SOURCE_DIR}/SQLiteCpp/src/*.cpp")
# file(GLOB_RECURSE SQLITECPP_SRC "${PROJECT_SOURCE_DIR}/SQLiteCpp/include/SQLiteCpp/*.h")
# add sources of the wrapper as a "SQLiteCpp" static library
# add_library(SQLiteCpp ${SQLITECPP_SRC})

#if (UNIX AND (CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
#    set_target_properties(SQLiteCpp PROPERTIES COMPILE_FLAGS "-fPIC")
#endif (UNIX AND (CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))

