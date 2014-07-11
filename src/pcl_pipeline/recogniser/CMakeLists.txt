cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(recogniser)
find_package(PCL 1.7)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1024") 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

include_directories(${PCL_INCLUDE_DIRS}, /Users/Michael/builds/pcl_and_dep/pcl/recognition/include/pcl/recognition/3rdparty/)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(recognise recognise.cpp render.cpp)
add_executable(global recognise_global.cpp my_prism_segment.cpp)

target_link_libraries(recognise ${PCL_IO_LIBRARIES}  ${PCL_LIBRARIES})
target_link_libraries(global ${PCL_IO_LIBRARIES}  ${PCL_LIBRARIES})
