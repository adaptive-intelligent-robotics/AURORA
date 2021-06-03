cmake_minimum_required(VERSION 3.10)

project(Aurora)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_definitions(USE_KDTREE)
add_compile_definitions(GRAPHIC)
add_compile_definitions(ENVIRONMENT=\"hexa_cam_vertical\")
add_compile_definitions(ALGORITHM=\"aurora_curiosity\")
add_compile_definitions(LATENT_SPACE_SIZE=2)
add_compile_definitions(USE_COLORS)
add_compile_definitions(USE_VIDEOS)
add_compile_definitions(ENCODER_TYPE=\"cnn_ae\")

include_directories(/Users/looka/git/sferes2/exp/aurora/singularity/aurora.sif/workspace/include)
include_directories(/Users/looka/git/sferes2/exp/aurora/singularity/aurora.sif/usr/include/eigen3)
include_directories(/Users/looka/git/sferes2/exp/aurora/singularity/aurora.sif/usr/local/include)
include_directories(/Users/looka/git/sferes2/exp/aurora/singularity/aurora.sif/git/sferes2)
include_directories(/Users/looka/git/sferes2/exp/aurora/include)

set(Torch_DIR /Users/looka/git/sferes2/exp/aurora/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED /Users/looka/git/sferes2/exp/aurora/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Aurora cpp/aurora.cpp cpp/visualise.cpp cpp/project_latent.cpp cpp/compilation_variables.hpp)
target_link_libraries(Aurora "${TORCH_LIBRARIES}")
set_property(TARGET Aurora PROPERTY CXX_STANDARD 14)
