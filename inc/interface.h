#ifndef OPENVINO_DEMO_INTERFACE_H
#define OPENVINO_DEMO_INTERFACE_H
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include <memory>
#include "utils.h"


class Interface
{
public:
    Interface(std::string type, const std::string& modelPath, bool isGPU, cv::Size inputSize,
              int descriptor_width, int descriptor_height, int descriptor_dim, bool gray=false);
    ~Interface();
    void run(cv::Mat& image, cv::Mat& score_map_mat, cv::Mat& descriptor_map_mat);
private:

    void preprocessing(cv::Mat& image, float* &input);
    void postprocessing(float* buffer,
                        cv::Mat& descriptor_map_mat);


    nvinfer1::ICudaEngine* engine;
    nvinfer1::IRuntime* runtime;
    nvinfer1::IExecutionContext* context;
    std::unique_ptr<nvinfer1::IHostMemory> trtModelStream;
    cudaStream_t stream;

    cv::Size2f inputImageShape;
    bool grayscale = false;
    std::string model_type;

    void *buffers[3];
    float *tmp_buffer;

    int descriptor_width;
    int descriptor_height;
    int descriptor_dim;
    float *desc_tmp_buffer;
};

#endif //OPENVINO_DEMO_INTERFACE_H
