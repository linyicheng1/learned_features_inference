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
    int score_height = score_map_mat.rows;
    int score_width = score_map_mat.cols;
    memcpy((uchar*)score_map_mat.data, out1.data, score_height*score_width*sizeof(float));

    int desc_height = descriptor_map_mat.rows;
    int desc_width = descriptor_map_mat.cols;
    int desc_channels = descriptor_map_mat.channels();
    // chw -> hwc
    std::vector<cv::Mat> chw(desc_channels);
    for (int i = 0; i < desc_channels; i++) {
        chw[i] = cv::Mat(desc_height, desc_width, CV_32FC1);
        memcpy((uchar*)chw[i].data, out2.channel(i).data, desc_height*desc_width*sizeof(float));
    }
    cv::merge(chw, descriptor_map_mat);
}

