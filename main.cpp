#include "interface.h"
#include <chrono>
#include "extractor.h"

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
int descriptor_dim = 64;
int descriptor_width = IMAGE_WIDTH;
int descriptor_height = IMAGE_HEIGHT;

// ./tensorrt_demo SuperPoint /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/weight/sp.trt /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/1.jpg /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/2.jpg
// ./tensorrt_demo alike /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/weight/Alike.trt /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/1.jpg /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/2.jpg
// ./tensorrt_demo d2net /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/weight/D2Net.trt /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/1.jpg /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/2.jpg
// ./tensorrt_demo disk /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/weight/disk.trt /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/1.jpg /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/2.jpg
// ./tensorrt_demo xfeat /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/weight/xfeat.trt /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/1.jpg /home/server/linyicheng/cpp_proj/learn_features/learned_features_inference/images/2.jpg

int main(int argc, char const *argv[])
{
    if (argc < 5)
    {
        std::cout<<"Usage: ./tensorrt_demo <model_type> <model_path> <image_0_path> <image_0_path>"<<std::endl;
        return -1;
    }
    // 1. load the model
    std::shared_ptr<Interface> net_ptr;
    std::string model_type = argv[1];
    if (model_type == "alike")
    {
        descriptor_dim = 64;
        descriptor_width = IMAGE_WIDTH;
        descriptor_height = IMAGE_HEIGHT;
        net_ptr = std::make_shared<Interface>("alike", argv[2], true,
                                              cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT),
                                              descriptor_width, descriptor_height, descriptor_dim);
    }
    else if (model_type == "d2net")
    {
        descriptor_dim = 512;
        descriptor_width = IMAGE_WIDTH / 8;
        descriptor_height = IMAGE_HEIGHT / 8;
        net_ptr = std::make_shared<Interface>("d2net", argv[2], true,
                                              cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT),
                                              descriptor_width, descriptor_height, descriptor_dim);
    }
    else if (model_type == "SuperPoint")
    {
        descriptor_dim = 256;
        descriptor_width = IMAGE_WIDTH / 8;
        descriptor_height = IMAGE_HEIGHT / 8;
        net_ptr = std::make_shared<Interface>("SuperPoint", argv[2], true,
                                              cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT),
                                              descriptor_width, descriptor_height, descriptor_dim);
    }
    else if (model_type == "disk")
    {
        descriptor_dim = 128;
        descriptor_width = IMAGE_WIDTH;
        descriptor_height = IMAGE_HEIGHT;
        net_ptr = std::make_shared<Interface>("disk", argv[2], true,
                                              cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT),
                                              descriptor_width, descriptor_height, descriptor_dim);
    }
    else if (model_type == "xfeat")
    {
        descriptor_dim = 64;
        descriptor_width = IMAGE_WIDTH / 8;
        descriptor_height = IMAGE_HEIGHT / 8;
        net_ptr = std::make_shared<Interface>("xfeat", argv[2], true,
                                              cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT),
                                              descriptor_width, descriptor_height, descriptor_dim);
    }
    else
    {
        std::cout<<"model type not supported"<<std::endl;
        return -1;
    }

    // 2. load the images
    std::string img_0_path = argv[3];
    std::string img_1_path = argv[4];
    cv::Mat image = cv::imread(img_0_path);
    cv::Mat image2 = cv::imread(img_1_path);
    cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    cv::resize(image2, image2, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

    // 3. run the model && extract the key points
    std::vector<cv::KeyPoint> key_points;
    cv::Mat score_map = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1), \
            desc_map = cv::Mat(descriptor_height, descriptor_width, CV_32FC(descriptor_dim));
    cv::Mat desc;

    std::vector<cv::KeyPoint> key_points2;
    cv::Mat score_map2 = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1), \
            desc_map2 = cv::Mat(descriptor_height, descriptor_width, CV_32FC(descriptor_dim));
    cv::Mat desc2;

    auto start = std::chrono::system_clock::now();
//    for (int i = 0;i < 500; i ++)
    {
        net_ptr->run(image, score_map, desc_map);
    }
    auto end = std::chrono::system_clock::now();

    // print the mean cost in ms
    std::cout<<"mean cost: "<<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f / 500.f <<" ms"<<std::endl;

    net_ptr->run(image, score_map, desc_map);
    key_points = nms(score_map, 500, 0.01, 16, cv::Mat());
    desc = bilinear_interpolation(image.cols, image.rows, desc_map, key_points);

    net_ptr->run(image2, score_map2, desc_map2);
    key_points2 = nms(score_map2, 500, 0.01, 16,cv::Mat());
    desc2 = bilinear_interpolation(image2.cols, image2.rows, desc_map2, key_points2);

    // 4. match the key points
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(desc, desc2, matches);

    // 5. fundamental matrix estimation
    std::vector<cv::Point2f> points1, points2;
    for (auto& match : matches)
    {
        points1.push_back(key_points[match.queryIdx].pt);
        points2.push_back(key_points2[match.trainIdx].pt);
    }
    std::vector<uchar> status;
    cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, status);

    std::vector<cv::DMatch> good_matches;
    std::vector<cv::DMatch> bad_matches;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == 1)
        {
            good_matches.push_back(matches[i]);
        }
        else
        {
            bad_matches.push_back(matches[i]);
        }
    }

    // 6. show the matches
    cv::Mat img_matches = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH * 2, CV_8UC3);
    image.copyTo(img_matches(cv::Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)));
    image2.copyTo(img_matches(cv::Rect(IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_HEIGHT)));

    for (const auto m:bad_matches) {
        cv::Point2f pt1 = key_points[m.queryIdx].pt;
        cv::Point2f pt2 = key_points2[m.trainIdx].pt;
        cv::line(img_matches, pt1, cv::Point2f(pt2.x + IMAGE_WIDTH, pt2.y), cv::Scalar(0, 0, 255), 2);
    }
    for (const auto m:good_matches) {
        cv::Point2f pt1 = key_points[m.queryIdx].pt;
        cv::Point2f pt2 = key_points2[m.trainIdx].pt;
        cv::line(img_matches, pt1, cv::Point2f(pt2.x + IMAGE_WIDTH, pt2.y), cv::Scalar(0, 255, 0), 2);
    }

//    cv::imshow("matches", img_matches);
    cv::imwrite("matches.jpg", img_matches);

    cv::Mat score_map_show = score_map * 255.;
    score_map_show.convertTo(score_map_show, CV_8UC1);
    cv::cvtColor(score_map_show, score_map_show, cv::COLOR_GRAY2BGR);

    cv::Mat score_map_show2 = score_map2 * 255.;
    score_map_show2.convertTo(score_map_show2, CV_8UC1);
    cv::cvtColor(score_map_show2, score_map_show2, cv::COLOR_GRAY2BGR);

    for (auto& kp : key_points)
    {
        cv::circle(score_map_show, kp.pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    for (auto& kp : key_points2)
    {
        cv::circle(score_map_show2, kp.pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    cv::imwrite("kps0.jpg", score_map_show);
    cv::imwrite("kps1.jpg", score_map_show2);
//    cv::waitKey(0);

    std::cout<<"mean cost: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f / 500.f <<" s"<<std::endl;
}
