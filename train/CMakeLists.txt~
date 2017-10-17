cmake_minimum_required(VERSION 2.8)
project(MTCNN_train)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp MTCNN.cpp MTCNN.h)
add_executable(MTCNN_train ${SOURCE_FILES})

target_link_libraries(MTCNN_train ${OpenCV_LIBS} )
target_link_libraries(MTCNN_train ${Caffe_LIBRARIES})
