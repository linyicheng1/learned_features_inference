#ifndef NCNN_DEMO_INTERFACE_H
#define NCNN_DEMO_INTERFACE_H
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include "utils.h"
#include "net.h"

class Interface
{
public:
    Interface(std::string type, const std::string& modelPath, bool isGPU, cv::Size inputSize, bool gray=false);
    void run(cv::Mat& image, cv::Mat& score_map_mat, cv::Mat& descriptor_map_mat);
private:

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    const float mean_vals_inv[3] = {0, 0, 0};
    const float norm_vals_inv[3] = {255.f, 255.f, 255.f};
    ncnn::Net net;
    ncnn::Mat in;
    ncnn::Mat out1, out2;
};

#endif //NCNN_DEMO_INTERFACE_H
