cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(recogniser)
find_package(PCL 1.7)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1024") 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${PCL_INCLUDE_DIRS}, /Users/Michael/builds/pcl_and_dep/pcl/recognition/include/pcl/recognition/3rdparty/)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(recognise recognise.cpp render.cpp local_recognizer.cpp)
add_executable(test vtk_test.cpp)

target_link_libraries(recognise ${PCL_IO_LIBRARIES}  ${PCL_LIBRARIES})
target_link_libraries(test ${PCL_IO_LIBRARIES}  ${PCL_LIBRARIES})