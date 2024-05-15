#include "interface.h"
#include <chrono>
#include "extractor.h"


int main(int argc, char const *argv[])
{
    Interface net{"alike", "/home/vio/Code/NN/openvino_d2net/Alike.xml", true, cv::Size(400,400)};
    cv::Mat image = cv::imread("../1.jpg");
    cv::Mat image2 = cv::imread("../2.jpg");
    cv::resize(image, image, cv::Size(512, 512));
    cv::resize(image2, image2, cv::Size(512, 512));
    
    std::vector<cv::KeyPoint> key_points;
    cv::Mat score_map = cv::Mat(512, 512, CV_32FC1), desc_map = cv::Mat(512, 512, CV_32FC(64));
    cv::Mat desc;

    std::vector<cv::KeyPoint> key_points2;
    cv::Mat score_map2 = cv::Mat(512, 512, CV_32FC1), desc_map2 = cv::Mat(512, 512, CV_32FC(64));
    cv::Mat desc2;

    auto start = std::chrono::system_clock::now();
    net.run(image, score_map, desc_map);
    key_points = nms(score_map, 500, 0.01, 16, cv::Mat());
    desc = bilinear_interpolation(image.cols, image.rows, desc_map, key_points);

    net.run(image2, score_map2, desc_map2);
    key_points2 = nms(score_map2, 500, 0.01, 16,cv::Mat());
    desc2 = bilinear_interpolation(image2.cols, image2.rows, desc_map2, key_points2);

    auto end = std::chrono::system_clock::now();

    // match the key points
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(desc, desc2, matches);

    cv::Mat img_matches;
    cv::drawMatches(image, key_points, image2, key_points2, matches, img_matches);
    cv::imshow("matches", img_matches);

    cv::Mat score_map_show = score_map * 255.;
    score_map_show.convertTo(score_map_show, CV_8UC1);
    cv::cvtColor(score_map_show, score_map_show, cv::COLOR_GRAY2BGR);

    cv::Mat score_map_show2 = score_map2 * 255.;
    score_map_show2.convertTo(score_map_show2, CV_8UC1);
    cv::cvtColor(score_map_show2, score_map_show2, cv::COLOR_GRAY2BGR);

    for (auto& kp : key_points)
    {
        cv::circle(image, kp.pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    for (auto& kp : key_points2)
    {
        cv::circle(image2, kp.pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    cv::imwrite("kps0.jpg", image);
    cv::imwrite("kps1.jpg", image2);
    cv::waitKey(0);

    std::cout<<"mean cost: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f <<" s"<<std::endl;
}
