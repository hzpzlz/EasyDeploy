#ifndef __ED_MNN_CLASS_H__
#define __ED_MNN_CLASS_H__

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include "ed_engine_base.h"
using namespace MNN;

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

    int checkAndCreateInputBuffer(const std::vector<EdNodeInfo> node_info,
                            std::map<std::string, EdBuffer> buffer);
    
    int checkAndCreateOutputBuffer(const std::vector<EdNodeInfo> node_info,
                            std::map<std::string, EdBuffer> buffer);

    std::shared_ptr<MNN::Interpreter> net;
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;
    MNN::Session *session;

    std::map<std::string, MNN::Tensor*> inputData;
    std::map<std::string, MNN::Tensor*> outputData;
    
    std::map<std::string, MNN::Tensor*> inputDataHost;
    std::map<std::string, MNN::Tensor*> outputDataHost;

    MNN::Tensor *inputDataHostItem;
    MNN::Tensor *outputDataHostItem;
};

#endif