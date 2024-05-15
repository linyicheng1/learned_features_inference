#ifndef OPENVINO_DEMO_UTILS_H
#define OPENVINO_DEMO_UTILS_H

#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>


struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

namespace utils
{
    size_t vectorProduct(const std::vector<int64_t>& vector);
    std::wstring charToWstring(const char* str);
    std::vector<std::string> loadNames(const std::string& path);
    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
                            const std::vector<std::string>& classNames,
        const cv::Scalar& color);



    void letterbox(const cv::Mat& image, cv::Mat& outImage,
                   const cv::Size& newShape,
                   const cv::Scalar& color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

    template <typename T>
    T clip(const T& n, const T& lower, const T& upper);
}

#endif //OPENVINO_DEMO_UTILS_H
