#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
using namespace MNN;

int main(void)
{
    std::string image_name = "../assets/1.bmp";
    std::string model_name = "../assets/seg.mnn";
    int forward = MNN_FORWARD_CUDA;  //MNN_FORWARD_CUDA MNN_FORWARD_CPU
    //int forward = MNN_FORWARD_OPENCL;

    //初始化参数
    int precision  = 2;
    int power      = 0;
    int memory     = 0;
    int threads    = 1;
    int MODEL_SIZE = 768;

    //使用opencv读取图像
    cv::Mat raw_image    = cv::imread(image_name.c_str());
    int raw_image_height = raw_image.rows;
    int raw_image_width  = raw_image.cols; 
    //将图像resize到模型大小
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(MODEL_SIZE, MODEL_SIZE));
    printf("input img size: %d %d \n", raw_image_height, raw_image_width);

    std::shared_ptr<MNN::Interpreter> net;
    net.reset(MNN::Interpreter::createFromFile(model_name.c_str()));
    char *version = (char *)net->getModelVersion();
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
    net->releaseModel();

    // preprocessing
    image.convertTo(image, CV_32FC3, 1.0);
    //image = image / 255.0f;
    std::cout<<"img channel: "<<image.channels()<<std::endl;

    // wrapping input tensor, convert nhwc to nchw    
    std::vector<int> dims{1,  MODEL_SIZE, MODEL_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();

    printf("create input size: %d \n", nhwc_Tensor->size());
    printf("create input shape: %d %d %d %d \n", nhwc_Tensor->shape()[0], nhwc_Tensor->shape()[1], nhwc_Tensor->shape()[2], nhwc_Tensor->shape()[3]);

    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "input";
    auto inputTensor  = net->getSessionInput(session, nullptr);

    printf("device input size: %d \n", inputTensor->size());
    printf("device input shape: %d %d %d %d \n", inputTensor->shape()[0], inputTensor->shape()[1], inputTensor->shape()[2], inputTensor->shape()[3]);

    inputTensor->copyFromHostTensor(nhwc_Tensor);

    printf("device input size after copy from host: %d \n", inputTensor->size());
    printf("device input shape after copy from host: %d %d %d %d \n", inputTensor->shape()[0], inputTensor->shape()[1], inputTensor->shape()[2], inputTensor->shape()[3]);

    cv::Mat  seg_res1(MODEL_SIZE, MODEL_SIZE, CV_32FC3, nhwc_data);
    cv::imwrite("../assets/input2.jpg", seg_res1);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "output";

    MNN::Tensor *out_tensor  = net->getSessionOutput(session, output_tensor_name0.c_str());

    MNN::Tensor out_tensor_host(out_tensor, out_tensor->getDimensionType());
    
    if  (out_tensor->getDimensionType() == MNN::Tensor::TENSORFLOW) {
        std::cout<<"using tpye TENSORFLOW"<<std::endl;
    }
    else if  (out_tensor->getDimensionType() == MNN::Tensor::CAFFE) {
        std::cout<<"using tpye CAFFE"<<std::endl;
    }
    std::cout<<"device out_tensor tpye: "<<out_tensor->getDimensionType()<<std::endl;
    std::cout<<"device out_tensor size: "<<out_tensor->size()<<std::endl;
    std::cout<<"device out_tensor elesize: "<<out_tensor->elementSize()<<std::endl;
    
    out_tensor->copyToHostTensor(&out_tensor_host);
    std::cout<<"host tensor size: "<<out_tensor_host.size()<<std::endl;

    auto out_data = out_tensor_host.host<float>();

    cv::Mat  seg_res(MODEL_SIZE, MODEL_SIZE, CV_32FC1, cv::Scalar(255)); 
    ::memcpy(seg_res.data, out_data, MODEL_SIZE * MODEL_SIZE * sizeof(float));

    //cv::Mat  seg_res(MODEL_SIZE, MODEL_SIZE, CV_32FC1, out_data); 

    cv::imwrite("../assets/seg.jpg", seg_res * 255.0);

    return 0;
}


