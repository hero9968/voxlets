cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(recogniser)
find_package(PCL 1.7)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(recognise recognise.cpp)
target_link_libraries(recognise ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})