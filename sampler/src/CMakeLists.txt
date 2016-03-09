include(CheckLibraryExists)
include(AddCompilerFlag)
include(FindPkgConfig)

include(GetGitRevisionDescription)
git_describe(VERSION_MAJOR)

string(TIMESTAMP BUILD_DATE)

find_package(GSL REQUIRED)

# set up include paths
include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    )
include_directories(SYSTEM
    "${GSL_INCLUDES}"
    )

set(SAMPLER_LIBRARIES ${SAMPLER_LIBRARIES}
   ${GSL_LIBRARY}
   ${GSL_CBLAS_LIBRARY}
   )

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/pdg05.dat
                                            ${CMAKE_BINARY_DIR}/pdg05.dat
                                            )

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/chemical_potential.dat
                                            ${CMAKE_BINARY_DIR}/chemical_potential.dat)


set (SRC_LIST 
              sampler.cc
              ars.cc
              random.cc
              distributions.cc
              fourvector.cc
              inputfunctions.cc)

include_directories( ${CMAKE_SOURCE_DIR}/include )


set(SANITIZER_FLAG -fsanitize=address)

add_executable(main main.cc ${SRC_LIST})
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#add_library(sampler STATIC ${SRC_LIST})

#set_target_properties(main PROPERTIES
#    COMPILE_FLAGS "${SANITIZER_FLAG} -DBUILD_TESTS"
#         LINK_FLAGS ${SANITIZER_FLAG} 
#         )

target_link_libraries ( main ${SAMPLER_LIBRARIES} )
