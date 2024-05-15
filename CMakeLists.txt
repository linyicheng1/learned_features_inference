cmake_minimum_required(VERSION 3.10)
project(learned_features_inference_ncnn)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
set(ncnn_DIR "/home/vio/libs/ncnn/install/lib/cmake/ncnn/" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} inc)

add_executable(ncnn_demo main.cpp src/interface.cpp src/extractor.cpp src/utils.cpp)
target_link_libraries(ncnn_demo
        ncnn
        ${OpenCV_LIBS}
        -fopenmp
        )
