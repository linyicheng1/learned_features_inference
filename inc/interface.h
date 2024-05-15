#ifndef OPENVINO_DEMO_INTERFACE_H
#define OPENVINO_DEMO_INTERFACE_H
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include "openvino/openvino.hpp"
#include "utils.h"

class Interface
{
public:
    Interface(std::string type, const std::string& modelPath, bool isGPU, cv::Size inputSize, bool gray=false);
    void run(cv::Mat& image, cv::Mat& score_map_mat, cv::Mat& descriptor_map_mat);
private:
    void printInputAndOutputsInfoShort(const ov::Model& network);
    void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    void postprocessing(ov::InferRequest &score_map,
                        cv::Mat& score_map_mat,
                        cv::Mat& descriptor_map_mat);
    void d2net_postprocessing(ov::InferRequest &infer_request,
                              cv::Mat& score_map_mat,
                              cv::Mat& descriptor_map_mat);
    void alike_postprocessing(ov::InferRequest &infer_request,
                              cv::Mat& score_map_mat,
                              cv::Mat& descriptor_map_mat);
    ov::Core ie;
    std::shared_ptr<ov::Model> network;
    ov::CompiledModel executable_network;
    cv::Size2f inputImageShape;
    bool grayscale = false;
    std::string model_type;
};

#endif //OPENVINO_DEMO_INTERFACE_H
