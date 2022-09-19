#pragma once

#ifndef __ED_NCNN_CLASS_H__
#define __ED_NCNN_CLASS_H__

#include "ed_engine_base.h"
#include "net.h"

int setRuntime(int runtime, EdModelInfo& model_info);
int setPrecision(int precision, EdModelInfo& model_info);
int setPower(int power, EdModelInfo& model_info);
int setMemory(int memory, EdModelInfo& model_info);

class InferEngine {
public:
    InferEngine();
    ~InferEngine();
      
    int Init(EdModelInfo& model_info,
            std::string model_path,
            std::map<std::string, EdBuffer> input_data,
            std::map<std::string, EdBuffer> output_data);
    int Inference();

private:
    void setConfig(const EdModelInfo& model_info);

    // int checkAndCreateInputBuffer(const std::vector<EdNodeInfo> node_info,
    //                         std::map<std::string, EdBuffer> buffer);
    
    // int checkAndCreateOutputBuffer(const std::vector<EdNodeInfo> node_info,
    //                         std::map<std::string, EdBuffer> buffer);

    int checkIOBuffer(const std::vector<EdNodeInfo> in_node_info, std::map<std::string, EdBuffer> in_buffer,
                    const std::vector<EdNodeInfo> out_node_info, std::map<std::string, EdBuffer> out_buffer);
    ncnn::Net net;
    //ncnn::Extractor ex;

    // std::map<std::string, ncnn::Mat> inputData;
    // std::map<std::string, ncnn::Mat> outputData;

    // ncnn::Mat inputDataHostItem;
    // ncnn::Mat outputDataHostItem;

};


#endif