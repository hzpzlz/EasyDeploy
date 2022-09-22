#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "inference.h"
#include "post_process.h"

int main(void)
{
    std::string assets = "../assets";
    std::string img_path = "../assets/segmentation/1.bmp";
    std::string yaml_path = "../assets/segmentation/segmentation.yaml";

    int input_height = 768;
    int input_width = 768;
    int input_channel = 3;

    int output_height = 768;
    int output_width = 768;
    int output_channel = 1;

    InferConfig config;
    config.assets_path = assets;
    config.yaml_path = yaml_path;

    InferModelConfig model_config;
    model_config.runtime = 0;
    model_config.precision = 0;
    model_config.power = 0;
    model_config.memory = 0;
    
    //使用opencv读取图像
    cv::Mat raw_image    = cv::imread(img_path.c_str());
    int raw_image_height = raw_image.rows;
    int raw_image_width  = raw_image.cols; 
    //将图像resize到模型大小
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(input_height, input_width));
    printf("input img size: %d %d \n", raw_image_height, raw_image_width);

    // image preprocessing
#ifdef __MNN_INFER_FRAMEWORK__
    image.convertTo(image, CV_32FC3, 1.0);
    //image = image / 255.0;
#endif
    cv::Mat seg_res(output_height, output_width, CV_32FC1, cv::Scalar(255)); 

    model_config.input["input"].dataType = 32;
    model_config.input["input"].bufferSize = 1 * input_height * input_width * input_channel * sizeof(float);
    model_config.input["input"].bufferData = (void *)image.data;

    model_config.output["output"].dataType = 32;
    model_config.output["output"].bufferSize = 1 * output_height * output_width * output_channel * sizeof(float);
    model_config.output["output"].bufferData = (void *)seg_res.data;

    inference *infer = new inference();
    infer->Init(config, model_config);
    infer->process();

    cv::imwrite("../assets/segmentation/seg.jpg", seg_res * 255.0);
    printf("save seg output success!!!\n");

    return 0;
}


