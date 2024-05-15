#ifndef OPENVINO_DEMO_EXTRACTOR_H
#define OPENVINO_DEMO_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::KeyPoint> nms(cv::InputArray score_map, int maxCorners,
                              double qualityLevel, double minDistance, cv::InputArray _mask);
cv::Mat bilinear_interpolation(int image_w, int image_h, const cv::Mat& desc_map, const std::vector<cv::KeyPoint>& kps);

#endif //OPENVINO_DEMO_EXTRACTOR_H
