#include "interface.h"
#include <utility>

Interface::Interface(std::string type, const std::string& modelPath, bool isGPU, cv::Size inputSize, bool gray) {
    std::string param = modelPath + ".param";
    std::string bin = modelPath + ".bin";
    net.load_param(param.c_str());
    net.load_model(bin.c_str());
}

/**
 * @brief Run the network
 * @param image
 */
void Interface::run(cv::Mat &image, cv::Mat& score_map_mat, cv::Mat& descriptor_map_mat) {
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // opencv to ncnn
    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
    in.substract_mean_normalize(mean_vals, norm_vals);
    // extract
    ex.input("input", in);
    ex.extract("score", out1);
    ex.extract("descriptor", out2);
    // ncnn to opencv
    out1.substract_mean_normalize(mean_vals_inv, norm_vals_inv);
    out2.substract_mean_normalize(mean_vals_inv, norm_vals_inv);

    int score_height = score_map_mat.rows;
    int score_width = score_map_mat.cols;
    memcpy((uchar*)score_map_mat.data, out1.data, score_height*score_width*sizeof(float));

    int desc_height = descriptor_map_mat.rows;
    int desc_width = descriptor_map_mat.cols;
    int desc_channels = descriptor_map_mat.channels();
    cv::Mat desc_tmp(desc_height, desc_width, CV_32FC(desc_channels));
    memcpy((uchar*)desc_tmp.data, out2.data, desc_height*desc_width*desc_channels*sizeof(float));
}

