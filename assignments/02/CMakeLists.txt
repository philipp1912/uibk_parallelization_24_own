cmake_minimum_required(VERSION 3.9)

project(assignments C CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -fsanitize=undefined -Wall -Wextra -pedantic")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(mandelbrot_seq mandelbrot_seq.cpp)

add_executable(mandelbrot_mpi mandelbrot_mpi.cpp)
target_link_libraries(mandelbrot_mpi PUBLIC MPI::MPI_C)

add_executable(mandelbrot_mpi_mp mandelbrot_mpi_mp.cpp)
target_link_libraries(mandelbrot_mpi_mp PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(mandelbrot_mpi_mp PUBLIC MPI::MPI_C)

