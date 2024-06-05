cmake_minimum_required(VERSION 3.10)
project(learned_features_inference_openvino)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
# you need to set OpenVINO_DIR to the path where you installed OpenVINO
set(OpenVINO_DIR /home/vio/Code/NN/openvino/l_openvino_toolkit_ubuntu20_2024.0.0.14509.34caeefd078_x86_64/runtime/cmake)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenVINO version: ${OpenVINO_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS} inc)

add_executable(openvino_demo main.cpp src/interface.cpp src/extractor.cpp src/utils.cpp)

target_link_libraries(openvino_demo openvino::runtime ${OpenCV_LIBS})


