cmake_minimum_required(VERSION 3.16)
project(ShallowWaterKokkos LANGUAGES CXX)

set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CUDAToolkit_rt_LIBRARY "/usr/lib/x86_64-linux-gnu/librt.so.1" CACHE FILEPATH "" FORCE)

find_package(Kokkos REQUIRED)
find_package(CUDAToolkit QUIET)

foreach(_kokkos_target kokkos kokkoscore kokkoscontainers kokkosalgorithms kokkossimd)
  if(TARGET Kokkos::${_kokkos_target})
    get_target_property(_links Kokkos::${_kokkos_target} INTERFACE_LINK_LIBRARIES)
    if(_links)
      list(REMOVE_ITEM _links "/usr/lib/x86_64-linux-gnu/librt.a")
      list(FIND _links "rt" _rt_index)
      if(_rt_index EQUAL -1)
        list(APPEND _links "rt")
      endif()
      set_target_properties(Kokkos::${_kokkos_target} PROPERTIES INTERFACE_LINK_LIBRARIES "${_links}")
    endif()
  endif()
endforeach()

add_executable(shallow_water_pipeline shallow_water.cpp)
target_link_libraries(shallow_water_pipeline PRIVATE Kokkos::kokkos)

add_executable(shallow_water_spmv shallow_water_spmv.cpp)
target_link_libraries(shallow_water_spmv PRIVATE Kokkos::kokkos)

if(CUDAToolkit_FOUND AND TARGET CUDA::cusparse AND TARGET CUDA::cudart)
  add_executable(shallow_water_cublas_spmv shallow_water_cublas_spmv.cpp)
  target_link_libraries(shallow_water_cublas_spmv PRIVATE CUDA::cusparse CUDA::cudart)
  target_compile_features(shallow_water_cublas_spmv PRIVATE cxx_std_17)
  target_compile_options(shallow_water_cublas_spmv PRIVATE
    -gencode=arch=compute_70,code=\[sm_70,compute_70\]
    -lineinfo
    -Xptxas=-v
    --keep)
else()
  message(STATUS "Skipping shallow_water_cublas_spmv target (CUDA toolkit with cuSPARSE not found)")
endif()
