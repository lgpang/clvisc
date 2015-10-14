project(spec)

cmake_minimum_required(VERSION 2.8)

#cl_spec.cpp  EosPCEv0.cl  helper.h  kernel_ideal.cl
set ( SRC_LIST 
    ${PROJECT_SOURCE_DIR}/src/main.cpp 
    ${PROJECT_SOURCE_DIR}/src/cl_spec.cpp 
    ${PROJECT_SOURCE_DIR}/src/helper.h 
    ${PROJECT_SOURCE_DIR}/src/kernel_spec.cl 
    ${PROJECT_SOURCE_DIR}/src/kernel_decay.cl 
    ${PROJECT_SOURCE_DIR}/inc/cl_spec.h)
#aux_source_directory(src/ SRC_LIST)
#aux_source_directory(inc/ SRC_LIST)
set( CMAKE_CXX_COMPILER  "g++" )

set ( CMAKE_USE_RELATIVE_PATH "true")
set ( CMAKE_EXECUTABLE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/ )
set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/ )

# 1D Block size used for 3D hydro, 4 halo cells ( i=0,1,BSZ-2,BSZ-1 )
set ( BSZ 256 )   

set ( DUMPDATA_FROM_GPU "true" )

set ( SWITCH_OFF_DECAY "false" )

option ( USE_DEVICE_GPU "Use GPU parallel" on )
option ( USE_SINGLE_PRECISION "Use float instead of double" on )
option ( CL_USE_DEPRECATED_OPENCL_1_1_APIS "Use Nvidia GPU" on )

option ( LOEWE_CSC "Use LOEWE AMD GPUs for spec kernel" off )


configure_file (
   ${PROJECT_SOURCE_DIR}/inc/Config.h.in
   ${PROJECT_SOURCE_DIR}/inc/Config.h
)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../FindOpenCL/")
 
find_package( OpenCL REQUIRED )
 
 
if( OPENCL_HAS_CPP_BINDINGS )
    message( "OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS} )
else( OPENCL_HAS_CPP_BINDINGS )
    message( "No OpenCL CPP bindings found" )
endif( OPENCL_HAS_CPP_BINDINGS )
 
include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories( "/usr/local/include/" )
link_directories( "/usr/local/lib" )


set ( EXTRA_LIBS ${EXTRA_LIBS} ${OPENCL_LIBRARIES} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories( ${PROJECT_SOURCE_DIR}/inc )

#include_directories( ${PROJECT_SOURCE_DIR}/../ReadConfig/inc )
#LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/../ReadConfig/lib )


add_executable ( spec ${SRC_LIST} )
target_link_libraries ( spec ${EXTRA_LIBS} )

