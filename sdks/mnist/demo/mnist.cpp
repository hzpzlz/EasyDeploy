#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#include "inference.h"
#include "post_process.h"

int main(void)
{
    std::string assets = "../assets";
    std::string model_name = "mnist.mnn";
    std::string img_path = "../assets/2.jpeg";
    
    int input_height = 28;
    int input_width = 28;
    int input_channel = 1;

    int output_height = 1;
    int output_width = 1;
    int output_channel = 10;

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
    cv::resize(raw_image, image, cv::Size(input_height, input_width));
    printf("input img size: %d %d \n", raw_image_height, raw_image_width);

    // image preprocessing
#ifdef __MNN_INFER_FRAMEWORK__
    image.convertTo(image, CV_32FC1, 1.0);
    image = image / 255.0;
#endif
    float *out_data = new float[output_channel];

    model_config.input["0"].dataType = 32;
    model_config.input["0"].bufferSize = 1 * input_height * input_width * input_channel * sizeof(float);
    model_config.input["0"].bufferData = (void *)image.data;

    model_config.output["32"].dataType = 32;
    model_config.output["32"].bufferSize = 1 * 1 * 1 * 10 * sizeof(float);
    model_config.output["32"].bufferData = (void *)out_data;

    inference *infer = new inference();
    infer->Init(config, model_config);
    infer->process();

    std::vector<float> cls_scores(output_channel, 0);

    for (int j = 0; j < output_channel; j++) {
        cls_scores[j] = out_data[j];
    }

    // softmax
    float exp_sum = 0.0f;
    for (int i = 0; i < 10; ++i)
    {
        float val = cls_scores[i];
        exp_sum += val;
    }
    // get result idx
    int  idx = 0;
    float max_prob = -10.0f;
    for (int i = 0; i < 10; ++i)
    {
        float val  = cls_scores[i];
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


