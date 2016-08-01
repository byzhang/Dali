cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("dali")
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET (CMAKE_EXPORT_COMPILE_COMMANDS 1)
set (CMAKE_EXPORT_COMPILE_COMMANDS "${CMAKE_EXPORT_COMPILE_COMMANDS}" CACHE STRING "export compile_commands.json")

if( UNIX AND NOT APPLE )
    execute_process(COMMAND /usr/bin/lsb_release -i -s
                            OUTPUT_VARIABLE LINUX_DISTRO
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

################################################################################
##                            COMPILER SETUP                                  ##
################################################################################

# ensure correct GCC version
set(GCC_MINIMUM_REQUIRED "4.9.2")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${GCC_MINIMUM_REQUIRED})
    message(FATAL_ERROR "Mimimum rquired version of gcc is ${GCC_MINIMUM_REQUIRED}")
  endif()
endif()

if (APPLE)
    set (CMAKE_CXX_COMPILER "clang++")
    set (MAYBE_SHARED "")
else(APPLE)
    # land of Linux or Windows
    set (MAYBE_SHARED "SHARED")
endif(APPLE)

if(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")
else(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w -fPIC")
endif(CMAKE_BUILD_TYPE MATCHES "debug")

if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "default")
endif (CMAKE_BUILD_TYPE STREQUAL "")
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})


################################################################################
##                            PACKAGES                                        ##
################################################################################
if( UNIX AND NOT APPLE AND LINUX_DISTRO STREQUAL Fedora)
    set(WITH_BLAS "open")
endif()

# REQUIRED PACKAGES
find_package(ZLIB REQUIRED)
find_package(BLAS REQUIRED)

# OPTIONAL PACKAGES
find_package(OpenBlas)
find_package(GooglePerfTools)
find_package(MKL)
find_package(CUDA)
# (Note: in certain cases BLAS is found
# by detecting Atlas, so we find this package
# separately again to obtain its dependencies)
find_package(Atlas)

IF (OpenBLAS_FOUND)
  message(STATUS "openblas found")
ENDIF (OpenBLAS_FOUND)

IF (ATLAS_FOUND)
  message(STATUS "Atlas found")
  # add specific Atlas libraries to dependencies
  LIST(APPEND BLAS_LIBRARIES ${Atlas_LIBRARIES})
ENDIF(ATLAS_FOUND)

if (NOT GPERFTOOLS_FOUND)
    message(WARNING "consider installing gperftools for better multithreaded performance.\n    > brew install gperftools")
endif(NOT GPERFTOOLS_FOUND)


# CLEANING UP AFTER FINDERS
# some library list variables are FALSE by default if libs are not found,
# but it is much more useful to keep it empty.
set(LIBVARS_TO_FIX MKL_LIBRARIES BLAS_LIBRARIES GPERFTOOLS_TCMALLOC_AND_PROFILER OpenBLAS_LIB)
foreach(libvar ${LIBVARS_TO_FIX})
    if (NOT ${libvar})
        set(${libvar} "")
    endif (NOT ${libvar})
endforeach(libvar)


# mshadow
set(MSHADOW_USE_CBLAS 1)
set(MSHADOW_USE_MKL 0)

# INCLUDES
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/mshadow)
include_directories(${MKL_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/third_party/libcuckoo/src)

if (GPERFTOOLS_FOUND)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
endif (GPERFTOOLS_FOUND)

################################################################################
##                            CUDA SETUP                                      ##
################################################################################

# CUDA BEGINS HERE
if (DEFINED WITH_CUDA)
    message(STATUS "Cuda support: ${WITH_CUDA}")
else()
    if (CUDA_FOUND STREQUAL TRUE)
        SET(WITH_CUDA TRUE)
    else()
        SET(WITH_CUDA FALSE)
        message(WARNING "CUDA not found - did you know that Dali has GPU support?")
    endif()
endif(DEFINED WITH_CUDA)


if (WITH_CUDA)
    SET(DALI_USE_CUDA 1)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    LIST(APPEND CUDA_NVCC_FLAGS --compiler-options ${CUDA_EXTRA_FLAGS} -lineinfo -Xptxas -dlcm=cg  -use_fast_math -std=c++11)
    SET(CUDA_EXTRA_FLAGS "-fno-strict-aliasing")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda/lib")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_EXTRA_FLAGS}")
    SET(MSHADOW_USE_CUDA 1)
else (WITH_CUDA)
    SET(MSHADOW_USE_CUDA 0)
endif (WITH_CUDA)


################################################################################
##                            TESTING SETUP                                   ##
################################################################################
##################################
# Download and install GoogleTest

include(ExternalProject)
ExternalProject_Add(gtest
  URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
  # Comment above line, and uncomment line below to use subversion.
  # SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk/
  # Uncomment line below to freeze a revision (here the one for 1.7.0)
  # SVN_REVISION -r700

  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(gtest source_dir binary_dir)

# This monstrosity came to live to alleviate two conditions:
#      1) ctest ignores colored output
#      2) make test does not build tests (wtf)
set(TESTS_MAIN "${PROJECT_SOURCE_DIR}/dali/tests_main.cpp")
function(add_testsuite TEST_TARGET TEST_DEPENDENCIES TEST_FILES)
    list(APPEND TEST_FILES ${TESTS_MAIN})
    if(GTEST_FOUND)
        add_executable(${TEST_TARGET} ${TEST_FILES})
        add_dependencies(${TEST_TARGET} ${TEST_DEPENDENCIES})
        target_link_libraries(${TEST_TARGET} ${TEST_DEPENDENCIES} ${binary_dir}/libgtest.a ${binary_dir}/libgtest_main.a)
        add_test(${TEST_TARGET} ${TEST_TARGET})
        add_dependencies(build_tests ${TEST_TARGET})
        add_custom_target(${TEST_TARGET}_run ${TEST_TARGET})
        add_dependencies(run_tests ${TEST_TARGET}_run)
    endif(GTEST_FOUND)
endfunction(add_testsuite)

SET(GTEST_FOUND 1)
if(GTEST_FOUND)
    enable_testing()
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    include_directories(${source_dir}/include)
    add_custom_target(run_tests true)
    add_custom_target(build_tests)
    add_dependencies(run_tests build_tests)
    ADD_CUSTOM_TARGET(run_tests_heapcheck
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/test_runner.sh ${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else(GTEST_FOUND)
    message(WARNING "GTESTS not found - install gtests to be able to run tests")
endif(GTEST_FOUND)


################################################################################
##                               DALI SETUP                                   ##
################################################################################

# Relative path in C++ are hard. Adding a macro:
set(DALI_DATA_DIR "${PROJECT_SOURCE_DIR}/data")

# ADDING DIRECTORIES WITH THEIR OWN CMAKELISTS FILES
add_subdirectory(${PROJECT_SOURCE_DIR}/dali)

# THOSE HEADERS ARE REQUIRED FOR DALI HEADERS TO WORK, SO WE AUTOMATICALLY INSTALL THEM.
# install mshadow
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/mshadow/mshadow DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    PATTERN "*.cuh")


################################################################################
##                            PACKAGE CREATION                                ##
################################################################################
# Generic CPACK setttings

set (CPACK_PACKAGE_NAME ${PROJECT_NAME})
set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set (CPACK_PACKAGE_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set (CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
set (CPACK_PACKAGE_DESCRIPTION "Buttery Smooth Automatic Differentiation in C++")

# name package according to device support
if (DALI_USE_CUDA)
    set(CPACK_XPU "gpu")
    set(CPACK_FORMULA_CLASSNAME "DaliGpu")
    set(CPACK_FORMULA_FNAME "dali-gpu")
else (DALI_USE_CUDA)
    set(CPACK_XPU "cpu")
    set(CPACK_FORMULA_CLASSNAME "DaliCpu")
    set(CPACK_FORMULA_FNAME "dali-cpu")
endif (DALI_USE_CUDA)


set (CPACK_SOURCE_GENERATOR "TGZ")
set (CPACK_SOURCE_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION})
set (CPACK_SOURCE_IGNORE_FILES "/data/;/docs/;/build/;/build2/;/build_2/;/build_cpu/;/.git/")

if ( UNIX AND NOT APPLE AND NOT WIN32 )
    message(STATUS "Package will be generated for Linux (DEB,RPM)")
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        set(CPACK_ARCHITECTURE "x86_64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    else()
        set(CPACK_ARCHITECTURE "i386")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
    endif()
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}-${CPACK_ARCHITECTURE}-${CPACK_XPU}")

    set(CPACK_PACKAGE_CONTACT "Szymon Sidor szymon.sidor@gmail.com")
    # DEBIAN
    list(APPEND CPACK_GENERATOR "DEB")
	set(CPACK_DEBIAN_PACKAGE_DEPENDS "zlib1g-dev,build-essential,libblas-dev,gcc-4.9,g++-4.9,cpp-4.9")

    # FEDORA
    list(APPEND CPACK_GENERATOR "RPM")
    set(CPACK_RPM_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_RPM_PACKAGE_SUMMARY ${CPACK_PACKAGE_DESCRIPTION})
    SET(CPACK_RPM_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
    SET(CPACK_RPM_PACKAGE_RELEASE "1")
    SET(CPACK_RPM_PACKAGE_LICENSE "MIT")
    SET(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")  # FROM: https://fedoraproject.org/wiki/RPMGroups
    SET(CPACK_RPM_PACKAGE_VENDOR "Dali Inc.")
    # TODO(szymon): add fedora dependencies
    #set(CPACK_RPM_PACKAGE_REQUIRES "python >= 2.5.0, cmake >= 2.8")

    include(CPack)
endif (UNIX AND NOT APPLE AND NOT WIN32 )

if ( APPLE )
    message(STATUS "Package will be generated for Mac OSX")
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}-${CPACK_XPU}")
    set(CPACK_PACKAGE_CONTACT "Jonathan Raiman jonathanraiman@gmail.com")
    SET(CPACK_GENERATOR "ZIP")

    add_custom_target(package_apple
        COMMAND "${CMAKE_CPACK_COMMAND}"
    "-C" "$<CONFIGURATION>" "--config" "${CMAKE_BINARY_DIR}/CpackConfig.cmake")

    add_custom_command(
        TARGET package_apple
        POST_BUILD
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/homebrew_dali_template.sh
                ${PROJECT_SOURCE_DIR}/misc/homebrew_dali_template.rb.in
                ${CMAKE_BINARY_DIR}/${CPACK_FORMULA_FNAME}.rb
                ${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.zip
                "CPACK_PACKAGE_FILE_NAME:${CPACK_PACKAGE_FILE_NAME}"
                "CPACK_PACKAGE_DESCRIPTION:${CPACK_PACKAGE_DESCRIPTION}"
                "CPACK_PACKAGE_VERSION:${CPACK_PACKAGE_VERSION}"
                "CPACK_FORMULA_CLASSNAME:${CPACK_FORMULA_CLASSNAME}"

    )
    include(CPack)
endif (APPLE)
