cmake_minimum_required(VERSION 3.5.1)

function(build_benchmark SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "benchmark_${src_name}")
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
endfunction(build_benchmark)

if (GTN_BUILD_BENCHMARKS)
  build_benchmark(${PROJECT_SOURCE_DIR}/benchmarks/ctc.cpp)
  build_benchmark(${PROJECT_SOURCE_DIR}/benchmarks/graph.cpp)
  build_benchmark(${PROJECT_SOURCE_DIR}/benchmarks/functions.cpp)
  build_benchmark(${PROJECT_SOURCE_DIR}/benchmarks/parallel.cpp)
endif ()

