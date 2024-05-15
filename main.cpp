#include "interface.h"
#include <chrono>
#include "extractor.h"

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
int descriptor_dim = 64;
int descriptor_width = IMAGE_WIDTH;
int descriptor_height = IMAGE_HEIGHT;

int main(int argc, char const *argv[])
{
    if (argc < 5)
    {
        std::cout<<"Usage: ./main <model_type> <model_path> <image_0_path> <image_0_path>"<<std::endl;
        return -1;
    }
    // 1. load the model
    std::shared_ptr<Interface> net_ptr;
    std::string model_type = argv[1];
    if (model_type == "alike")
    {
        net_ptr = std::make_shared<Interface>("alike", argv[2], true, cv::Size(512,512));
        descriptor_dim = 64;
        descriptor_width = IMAGE_WIDTH;
        descriptor_height = IMAGE_HEIGHT;
    }
    else if (model_type == "d2net")
    {
        net_ptr = std::make_shared<Interface>("d2net", argv[2], true, cv::Size(512,512));
        descriptor_dim = 512;
    }
    else if (model_type == "SuperPoint")
    {
        net_ptr = std::make_shared<Interface>("SuperPoint", argv[2], true, cv::Size(512,512));
        descriptor_dim = 256;
        descriptor_width = IMAGE_WIDTH / 8;
        descriptor_height = IMAGE_HEIGHT / 8;
    }
    else if (model_type == "disk")
    {
        net_ptr = std::make_shared<Interface>("disk", argv[2], true, cv::Size(512,512));
        descriptor_dim = 128;
        descriptor_width = IMAGE_WIDTH;
        descriptor_height = IMAGE_HEIGHT;
    }
    else if (model_type == "xfeat")
    {
        net_ptr = std::make_shared<Interface>("xfeat", argv[2], true, cv::Size(512,512));
        descriptor_dim = 64;
        descriptor_width = IMAGE_WIDTH / 8;
        descriptor_height = IMAGE_HEIGHT / 8;
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
    cv::Mat score_map = cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1), \
            desc_map = cv::Mat(descriptor_width, descriptor_height, CV_32FC(descriptor_dim));
    cv::Mat desc;

    std::vector<cv::KeyPoint> key_points2;
    cv::Mat score_map2 = cv::Mat(512, 512, CV_32FC1), \
            desc_map2 = cv::Mat(descriptor_width, descriptor_height, CV_32FC(descriptor_dim));
    cv::Mat desc2;

    auto start = std::chrono::system_clock::now();
    net_ptr->run(image, score_map, desc_map);
    key_points = nms(score_map, 500, 0.01, 16, cv::Mat());
    desc = bilinear_interpolation(image.cols, image.rows, desc_map, key_points);

    net_ptr->run(image2, score_map2, desc_map2);
    key_points2 = nms(score_map2, 500, 0.01, 16,cv::Mat());
    desc2 = bilinear_interpolation(image2.cols, image2.rows, desc_map2, key_points2);

    auto end = std::chrono::system_clock::now();

    // 4. match the key points
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(desc, desc2, matches);

    // 5. show the matches
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
