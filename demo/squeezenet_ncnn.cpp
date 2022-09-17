#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <algorithm>

#include "net.h"

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
    const char *ncnn_param = "../assets/squeezenet_v1.1.param";
    const char *ncnn_bin = "../assets/squeezenet_v1.1.bin";

    cv::Mat img = cv::imread("../assets/cat.jpeg");
    printf("img size: %d %d \n", img.rows, img.cols);

    ncnn::Net squeezenet;

    squeezenet.opt.use_vulkan_compute = true;

    squeezenet.load_param(ncnn_param);
    squeezenet.load_model(ncnn_bin);

    ncnn::Mat input_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 227, 227);

    //const float mean_values[3] = {104.f, 117.f, 123.f};
    const float mean_values[3] = {0.f, 0.f, 0.f};
    const float val_values[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input_img.substract_mean_normalize(mean_values, val_values);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.input("data", input_img);

    ncnn::Mat out;
    ex.extract("prob", out);

    std::vector<float> cls_scores;
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++) {
        cls_scores[j] = out[j];
    }

    print_topk(cls_scores, 3);

    return 0;
}


