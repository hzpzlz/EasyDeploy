#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "inference.h"
#include "post_process.h"

int main(void)
{
    std::string assets = "../assets";
    std::string model_name = "squeezenet_v1.1.mnn";
    std::string img_path = "../assets/cat.jpeg";
    
    int MODEL_SIZE = 227;

    InferConfig config;
    config.assets_path = assets;
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
    cv::resize(raw_image, image, cv::Size(MODEL_SIZE, MODEL_SIZE));
    printf("input img size: %d %d \n", raw_image_height, raw_image_width);

    // image preprocessing
    //const float mean_values[3] = {104.f, 117.f, 123.f};
    //const float val_values[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    image.convertTo(image, CV_32FC3, 1.0);
    image = image / 255.0;

    // std::map<std::string, EdBuffer> input;
    // std::map<std::string, EdBuffer> output;

    float *out_data = new float[1000];

    model_config.input["data"].dataType = 32;
    model_config.input["data"].bufferSize = 1 * MODEL_SIZE * MODEL_SIZE * 3 * sizeof(float);
    model_config.input["data"].bufferData = (void *)image.data;

    model_config.output["prob"].dataType = 32;
    model_config.output["prob"].bufferSize = 1 * 1 * 1 * 1000 * sizeof(float);
    model_config.output["prob"].bufferData = (void *)out_data;

    inference *infer = new inference();
    infer->Init(config, model_config);
    infer->process();

    std::vector<float> cls_scores(1000, 0);

    for (int j = 0; j < 1000; j++) {
        cls_scores[j] = out_data[j];
    }

    print_topk(cls_scores, 3);

    return 0;
}


