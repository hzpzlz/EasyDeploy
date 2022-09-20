#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "inference.h"
#include "post_process.h"

int main(void)
{
    std::string image_name = "../assets/2.jpeg";
    std::string model_name = "../assets/mnist.mnn";
    int forward = MNN_FORWARD_CPU;
    // int forward = MNN_FORWARD_OPENCL;

    int precision  = 2;
    int power      = 0;
    int memory     = 0;
    int threads    = 1;
    int INPUT_SIZE = 28;

    cv::Mat raw_image    = cv::imread(image_name.c_str());
    int raw_image_height = raw_image.rows;
    int raw_image_width  = raw_image.cols; 
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));

    // load and config mnn model
    // auto revertor = std::unique_ptr<Revert>(new Revert(model_name.c_str()));
    // revertor->initialize();
    // auto modelBuffer      = revertor->getBuffer();
    // const auto bufferSize = revertor->getBufferSize();
    // auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    // revertor.reset();

    std::shared_ptr<MNN::Interpreter> net;
    net.reset(MNN::Interpreter::createFromFile(model_name.c_str()));
    char * version = (char *)net->getModelVersion();
    printf("MNN version: %s \n", version);

    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    
    auto session = net->createSession(config);   

    // std::map<std::string, Tensor *> m_input;
    // m_input = net->getSessionInputAll(session);
    
    // for (auto iter:m_input) {
    //     auto name = iter.first;
    //     auto mnn_input_item = iter.second;
    //     auto shape = mnn_input_item->shape();
    //     printf("%s ssssssssss\n", std::string(name));
    // }

    net->releaseModel();

    clock_t start = clock();
    // preprocessing
    image.convertTo(image, CV_32FC3);
    image = image / 255.0f;

    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "0";
    auto inputTensor  = net->getSessionInput(session, nullptr);

    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "32";

    MNN::Tensor *tensor_scores  = net->getSessionOutput(session, output_tensor_name0.c_str());

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    
    tensor_scores->copyToHostTensor(&tensor_scores_host);

    // post processing steps
    auto scores_dataPtr  = tensor_scores_host.host<float>();

    // softmax
    float exp_sum = 0.0f;
    for (int i = 0; i < 10; ++i)
    {
        float val = scores_dataPtr[i];
        exp_sum += val;
    }
    // get result idx
    int  idx = 0;
    float max_prob = -10.0f;
    for (int i = 0; i < 10; ++i)
    {
        float val  = scores_dataPtr[i];
        //printf("%f \n", val);
        float prob = val / exp_sum;
        printf("%f \n", prob);
        if (prob > max_prob)
        {
            max_prob = prob;
            idx      = i;
        }
    }

    printf("the result is %d\n", idx);

    return 0;
}


