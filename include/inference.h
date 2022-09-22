//#pragma once

#ifndef __ED_INFERENCE_H__
#define __ED_INFERENCE_H__

#include <string>
#include <map>

#ifdef __MNN_INFER_FRAMEWORK__
#include "mnn_class.h"
#endif
#ifdef __NCNN_INFER_FRAMEWORK__
#include "ncnn_class.h"
#endif

struct InferConfig
{
    /* data */
    std::string assets_path;
    std::string yaml_path;
};

struct InferModelConfig
{
    /* data */
    std::string modelName;
    int runtime;
    int precision = 0;
    int power;
    int memory = 0;
    std::map<std::string, EdBuffer> input;
    std::map<std::string, EdBuffer> output;
};

class inference {
public:
    inference();
    ~inference();

    int Init(InferConfig config, InferModelConfig model_config);
    int process();


private:
    InferEngine *infer_engine = nullptr;
};

#endif