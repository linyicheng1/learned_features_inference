#include "interface.h"
#include <cuda_runtime_api.h>
#include <utility>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

Interface::Interface(std::string type, const std::string& modelPath, bool isGPU, cv::Size inputSize,
                     int _descriptor_width, int _descriptor_height, int _descriptor_dim, bool gray) {
    model_type = std::move(type);
    descriptor_width = _descriptor_width;
    descriptor_height = _descriptor_height;
    descriptor_dim = _descriptor_dim;

    // Load TensorRT model
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening file: " << modelPath << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(buffer.data(), size);
    context = engine->createExecutionContext();

    this->inputImageShape = cv::Size2f(inputSize);
    this->grayscale = gray;


    // Allocate memory for the input tensor
    if (grayscale) {
        cudaMalloc(&buffers[0], 1 * inputSize.width * inputSize.height * sizeof(float));
    } else {
        cudaMalloc(&buffers[0], 3 * inputSize.width * inputSize.height * sizeof(float));
    }

    // Allocate memory for the output tensor
    cudaMalloc(&buffers[1], inputSize.width * inputSize.height * sizeof(float));
    cudaMalloc(&buffers[2], descriptor_width * descriptor_height * descriptor_dim * sizeof(float));

    context->setTensorAddress("input", buffers[0]);
    context->setTensorAddress("score", buffers[1]);
    context->setTensorAddress("descriptor", buffers[2]);

    cudaStreamCreate(&stream);

    if (grayscale) {
        tmp_buffer = new float[inputSize.width * inputSize.height];
    } else {
        tmp_buffer = new float[inputSize.width * inputSize.height * 3];
    }
    desc_tmp_buffer = new float[descriptor_width * descriptor_height * descriptor_dim];
}

Interface::~Interface() {
    int inputIndex = 0;
    int scoreOutputIndex = 1;
    int descriptorOutputIndex = 2;

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[scoreOutputIndex]);
    cudaFree(buffers[descriptorOutputIndex]);

    delete engine;
    delete runtime;
    delete context;

    delete[] tmp_buffer;
}

/**
 * @brief Run the network
 * @param image
 */
void Interface::run(cv::Mat &image, cv::Mat& score_map_mat, cv::Mat& descriptor_map_mat) {

    this->preprocessing(image, tmp_buffer);

    // Copy input to device
    int width = (int)(this->inputImageShape.width);
    int height = (int)(this->inputImageShape.height);

    cudaMemcpy(buffers[0], tmp_buffer, 3 * width * height * sizeof(float), cudaMemcpyHostToDevice);

    context->enqueueV3(stream);

    // Copy output to host
    cudaMemcpy(score_map_mat.data, buffers[1], width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(desc_tmp_buffer, buffers[2], descriptor_width * descriptor_height * descriptor_dim * sizeof(float), cudaMemcpyDeviceToHost);

    postprocessing(desc_tmp_buffer, descriptor_map_mat);
}


/**
 * @brief Preprocess the image before feeding it to the network
 * @param image opencv image
 * @param blob
 * @param inputTensorShape
 */
void Interface::preprocessing(cv::Mat& image, float* &input) {
    cv::Mat resizedImage, floatImage;

    if (grayscale && image.channels() == 3) {
        // bgr -> gray
        cv::cvtColor(image, resizedImage, cv::COLOR_BGR2GRAY);
    } else if(!grayscale) {
        // bgr -> rgb
        cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    }

    // resize
    auto width = (int)(this->inputImageShape.width);
    auto height = (int)(this->inputImageShape.height);
    utils::letterbox(resizedImage, resizedImage, cv::Size(width, height),
                      cv::Scalar(114, 114, 114), false,
                      false, true, 32);
    // convert to float
    if (grayscale) {
        resizedImage.convertTo(floatImage, CV_32FC1, 1 / 255.0);
    } else {
        resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    }

    if (grayscale) {
        floatImage.copyTo(cv::Mat(floatImage.rows, floatImage.cols, CV_32FC1, input));
    } else {
        cv::Size floatImageSize{ floatImage.cols, floatImage.rows };
        // hwc -> chw
        std::vector<cv::Mat> chw(floatImage.channels());
        for (int i = 0; i < floatImage.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, input + i * floatImageSize.width * floatImageSize.height);
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
void Interface::postprocessing(float* request, cv::Mat& desc_map) {
    const size_t height = desc_map.rows;
    const size_t width = desc_map.cols;
    const size_t num_channels = desc_map.channels();
    // chw -> hwc
    std::vector<cv::Mat> chw(num_channels);
    for (int i = 0; i < num_channels; ++i)
    {
        chw[i] = cv::Mat((int)height, (int)width, CV_32FC1, request + i * height * width);
    }
    cv::merge(chw, desc_map);
}





