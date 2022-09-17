#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
using namespace MNN;

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(void)
{
    std::string image_name = "../assets/cat.jpeg";
    std::string model_name = "../assets/squeezenet_v1.1.mnn";
    int forward = MNN_FORWARD_CPU;  //MNN_FORWARD_CUDA MNN_FORWARD_CPU
    //int forward = MNN_FORWARD_OPENCL;

    //初始化参数
    int precision  = 2;
    int power      = 0;
    int memory     = 0;
    int threads    = 1;
    int MODEL_SIZE = 227;

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
    const float mean_values[3] = {104.f, 117.f, 123.f};
    const float val_values[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    image.convertTo(image, CV_32FC3, 1.0);
    image = image / 255.0;

    // for (int i = 0; i < MODEL_SIZE; i++) {
    //     for (int j = 0; j < MODEL_SIZE; j++) {
    //         image.at<cv::Vec3b>(i, j)[0] -= mean_values[0];
    //         image.at<cv::Vec3b>(i, j)[1] -= mean_values[1];
    //         image.at<cv::Vec3b>(i, j)[2] -= mean_values[2];
    //         image.at<cv::Vec3b>(i, j)[0] *= val_values[0];
    //         image.at<cv::Vec3b>(i, j)[1] *= val_values[1];
    //         image.at<cv::Vec3b>(i, j)[2] *= val_values[2];
    //     }
    // }

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

    std::string input_tensor = "data";
    auto inputTensor  = net->getSessionInput(session, nullptr);

    printf("device input size: %d \n", inputTensor->size());
    printf("device input shape: %d %d %d %d \n", inputTensor->shape()[0], inputTensor->shape()[1], inputTensor->shape()[2], inputTensor->shape()[3]);

    inputTensor->copyFromHostTensor(nhwc_Tensor);

    printf("device input size after copy from host: %d \n", inputTensor->size());
    printf("device input shape after copy from host: %d %d %d %d \n", inputTensor->shape()[0], inputTensor->shape()[1], inputTensor->shape()[2], inputTensor->shape()[3]);

    cv::Mat  seg_res1(MODEL_SIZE, MODEL_SIZE, CV_32FC3, nhwc_data);
    cv::imwrite("../assets/input2.jpg", seg_res1 * 255);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "prob";

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

    std::vector<float> cls_scores(out_tensor_host.elementSize(), 0);
    //cls_scores.resize(out_tensor_host.elementSize());

    for (int j = 0; j < out_tensor_host.elementSize(); j++) {
        cls_scores[j] = out_data[j];
    }

    print_topk(cls_scores, 3);

    return 0;
}


