#include "interface.h"

#include <utility>

Interface::Interface(std::string type, const std::string& modelPath, bool isGPU, cv::Size inputSize, bool gray) {
    model_type = std::move(type);
    std::vector<std::string> availableDevices = ie.get_available_devices();
    for (auto & availableDevice : availableDevices) {
        printf("supported device name : %s \n", availableDevice.c_str());
    }

    network = ie.read_model(modelPath);

    this-> printInputAndOutputsInfoShort(*network);

    auto out_size = network->outputs().size();
    auto in_size = network->input().get_shape();

    ov::preprocess::PrePostProcessor ppp(network);

    ppp.input().tensor().set_element_type(ov::element::f32);

    ppp.input().model().set_layout("NCHW");

    network = ppp.build();

    // -------- Loading a model to the device --------
    if (isGPU)
    {
        executable_network = ie.compile_model(network, "GPU");
        std::cout<<" load model to GPU "<<std::endl;
    }
    else
    {
        executable_network = ie.compile_model(network, "CPU");
        std::cout<<" load model to CPU "<<std::endl;
    }

    this->inputImageShape = cv::Size2f(inputSize);
    this->grayscale = gray;
}

/**
 * @brief Run the network
 * @param image
 */
void Interface::run(cv::Mat &image, cv::Mat& score_map_mat, cv::Mat& descriptor_map_mat) {
    ov::element::Type input_type = ov::element::f32;
    ov::Shape input_shape = { 1,3,512, 512 };
    if (grayscale) {
        input_shape = { 1,1,512, 512 };
    }

    float* blob = nullptr;

    std::vector<int64_t> inputTensorShape{ 1, 3, -1, -1 };
    if (grayscale) {
        inputTensorShape = { 1, 1, -1, -1 };
    }

    this->preprocessing(image, blob, inputTensorShape);

    // to tensor type
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, blob);// resize_img.ptr());

    // infer
    ov::InferRequest infer_request = executable_network.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // get score map and descriptor
    postprocessing(infer_request, score_map_mat, descriptor_map_mat);

    delete[] blob;
}

/**
 * @brief Print input and output information of the network
 * @param network
 */
void Interface::printInputAndOutputsInfoShort(const ov::Model &network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& input : network.inputs()) {
        std::cout << "    " << input.get_any_name() << " (node: " << input.get_node()->get_friendly_name()
                  << ") : " << input.get_element_type() << " / " << ov::layout::get_layout(input).to_string()
                  << std::endl;
    }

    std::cout << "Network outputs:" << std::endl;
    for (auto&& output : network.outputs()) {
        std::string out_name = "***NO_NAME***";
        std::string node_name = "***NO_NAME***";

        // Workaround for "tensor has no name" issue
        try {
            out_name = output.get_any_name();
        }
        catch (const ov::Exception&) {
        }
        try {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        }
        catch (const ov::Exception&) {
        }

        std::cout << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
                  << ov::layout::get_layout(output).to_string() << std::endl;
    }
}

/**
 * @brief Preprocess the image before feeding it to the network
 * @param image opencv image
 * @param blob
 * @param inputTensorShape
 */
void Interface::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
    cv::Mat resizedImage, floatImage;

    if (grayscale && image.channels() == 3) {
        // bgr -> gray
        cv::cvtColor(image, resizedImage, cv::COLOR_BGR2GRAY);
    } else if(!grayscale) {
        // bgr -> rgb
        cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    }

    // resize image to 512x512
//    cv::resize(resizedImage, resizedImage, cv::Size(512, 512));
    utils::letterbox(resizedImage, resizedImage, cv::Size(512, 512),
                      cv::Scalar(114, 114, 114), false,
                      false, true, 32);
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // convert to float
    if (grayscale) {
        resizedImage.convertTo(floatImage, CV_32FC1, 1 / 255.0);
    } else {
        resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    }
    // allocate memory for the blob
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];

    if (grayscale) {
        floatImage.copyTo(cv::Mat(floatImage.rows, floatImage.cols, CV_32FC1, blob));
    } else {
        cv::Size floatImageSize{ floatImage.cols, floatImage.rows };
        // hwc -> chw
        std::vector<cv::Mat> chw(floatImage.channels());
        for (int i = 0; i < floatImage.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
        }
        cv::split(floatImage, chw);
    }
}

/**
 * @brief Postprocess the output of the network
 * @param request
 * @param score_map_mat
 * @param descriptor_map_mat
 */
void Interface::postprocessing(ov::InferRequest &request,
                               cv::Mat& score_map_mat,
                               cv::Mat& descriptor_map_mat) {
    if (model_type == "d2net" || model_type == "alike" || model_type == "SuperPoint" || model_type == "disk" || model_type == "xfeat") {
        alike_postprocessing(request, score_map_mat, descriptor_map_mat);
    } else {
        std::cout<<" can not find the model type "<<model_type<<std::endl;
    }
    // grayscale
//    if (model_type == "SuperPoint") {
//        grayscale = true;
//    }
}

void Interface::d2net_postprocessing(ov::InferRequest &infer_request, cv::Mat &score_map_mat,
                                     cv::Mat &descriptor_map_mat) {
    ov::Tensor desc_map = infer_request.get_output_tensor();
    const auto& dims = desc_map.get_shape();
    const size_t height = dims[2];
    const size_t width = dims[3];
    const size_t num_channels = dims[1];

    descriptor_map_mat = cv::Mat((int)height, (int)width, CV_32FC1, desc_map.data<float>());

    std::cout<<"here is the output tensor "<<std::endl;
}

void Interface::alike_postprocessing(ov::InferRequest &infer_request, cv::Mat &score_map_mat,
                                     cv::Mat &descriptor_map_mat) {
    ov::Tensor score_map = infer_request.get_output_tensor(0);
    const auto& dims = score_map.get_shape();
    memcpy((uchar*)score_map_mat.data, score_map.data<float>(), score_map_mat.total() * score_map_mat.elemSize());

    ov::Tensor desc_map = infer_request.get_output_tensor(1);
    const size_t height = desc_map.get_shape()[2];
    const size_t width = desc_map.get_shape()[3];
    const size_t num_channels = desc_map.get_shape()[1];
    // chw -> hwc
    std::vector<cv::Mat> chw(num_channels);
    for (int i = 0; i < num_channels; ++i)
    {
        chw[i] = cv::Mat((int)height, (int)width, CV_32FC1, desc_map.data<float>() + i * height * width);
    }
    cv::merge(chw, descriptor_map_mat);
}

