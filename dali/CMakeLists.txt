set(DaliDir "${PROJECT_SOURCE_DIR}/dali")

file(GLOB_RECURSE DaliHeaders "${DaliDir}/*.h")
file(GLOB_RECURSE DaliSources "${DaliDir}/*.cpp")
file(GLOB_RECURSE DaliTemplates "${DaliDir}/*.cpp.template")
file(GLOB_RECURSE DaliCudaSources "${DaliDir}/*.cu")


if (WITH_CUDA)
    SET(OLD_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    SET(CMAKE_CXX_FLAGS "-O3 -w -fPIC ${CUDA_EXTRA_FLAGS}")
    cuda_add_library(dali_cuda ${DaliCudaSources})
    target_link_libraries(dali_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    SET(CMAKE_CXX_FLAGS ${OLD_CXX_FLAGS})
endif(WITH_CUDA)

cuda_compile(xx_test ${DaliCudaSources})
message("${xx_test}")

# Remove tests from compilation
file(GLOB_RECURSE DaliTests "${DaliDir}/tests.cpp")
list(REMOVE_ITEM DaliSources ${DaliTests} ${TESTS_MAIN})

foreach(header ${DaliTemplates})
    message(STATUS "Template instantiator will run for ${header}")
    ADD_CUSTOM_TARGET(dali_templates
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/python_template_instantiator.sh ${header}
        WORKING_DIRECTORY ${DaliDir}
        DEPENDS ${header})
endforeach(header)

add_library(dali
        ${DaliSources}
        ${DaliHeaders}
)

add_custom_target(fix_libiomp5
	COMMAND sudo sh ${PROJECT_SOURCE_DIR}/scripts/fix_libiomp5.sh
)

add_dependencies(dali dali_templates proto)

if (BLAS_FOUND AND MKL_FOUND)
    add_definitions(-DEIGEN_USE_MKL_VML
                    -DEIGEN_USE_BLAS)
    add_dependencies(dali fix_libiomp5)
endif (BLAS_FOUND AND MKL_FOUND)

target_link_libraries(dali ${ZLIB_LIBRARIES}
                           proto
                           SQLiteCpp
                           redox_static
                           sqlite3
                           json11static
                           ${BLAS_LIBRARIES}
                           ${HIREDIS_LIBRARIES}
                           ${MKL_LIBRARIES}
                           ${GFLAGS_LIBRARIES})
if (GPERFTOOLS_FOUND)
    target_link_libraries(dali ${GPERFTOOLS_LIBRARIES})
endif (GPERFTOOLS_FOUND)
if (WITH_CUDA)
    target_link_libraries(dali dali_cuda)
endif (WITH_CUDA)


add_testsuite("test_dali" "dali" "${DaliTests}")
