cmake_minimum_required(VERSION 3.5.1)

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_link_libraries(
    ${target}
    PRIVATE
    gtn
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
    )
  add_test(${target} ${target})
endfunction(build_test)

if (GTN_BUILD_TESTS)
  build_test(${PROJECT_SOURCE_DIR}/test/autograd_test.cpp)
  build_test(${PROJECT_SOURCE_DIR}/test/criterion_test.cpp)
  build_test(${PROJECT_SOURCE_DIR}/test/functions_test.cpp)
  build_test(${PROJECT_SOURCE_DIR}/test/graph_test.cpp)
  build_test(${PROJECT_SOURCE_DIR}/test/utils_test.cpp)
  build_test(${PROJECT_SOURCE_DIR}/test/rand_test.cpp)
endif ()
