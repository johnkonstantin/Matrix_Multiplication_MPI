cmake_minimum_required(VERSION 3.24)
project(opp_lab4 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")

find_package(MPI REQUIRED)

add_executable(opp_lab4 main.c)

target_link_libraries(opp_lab4 PUBLIC MPI::MPI_C)
