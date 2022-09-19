#include "mnn_class.h"
using namespace MNN;

int setRuntime(int runtime, EdModelInfo& model_info) {
    if (1 == runtime) {
        model_info.runtime = MNN_RUNTIME_CPU;
    }
    else if (2 == runtime) {
        model_info.runtime = MNN_RUNTIME_AUTO;
    }
    else if (3 == runtime) {
        model_info.runtime = MNN_RUNTIME_CUDA;
    }
    else if (4 == runtime) {
        model_info.runtime = MNN_RUNTIME_OPENCL;
    }
    else if (5 == runtime) {
        model_info.runtime = MNN_RUNTIME_VULKAN;
    }
    else {
        model_info.runtime = MNN_RUNTIME_AUTO;
    }
    return 0;
}

int setPrecision(int precision, EdModelInfo& model_info) {
    if (1 == precision) {
        model_info.precision = MNN_PRECISION_HIGH;
    }
    else if(2 == precision) {
        model_info.precision = MNN_PRECISION_LOW;
    }
    else {
        model_info.precision = MNN_PRECISION_NORMAL;
    }
    return 0;
}

int setPower(int power, EdModelInfo& model_info) {
    if (1 == power) {
        model_info.power = MNN_POWER_HIGH;
    }
    else if(2 == power) {
        model_info.power = MNN_POWER_LOW;
    }
    else {
        model_info.power = MNN_POWER_NORMAL;
    }
    return 0;
}

int setMemory(int memory, EdModelInfo& model_info) {
    if (1 == memory) {
        model_info.memory = MNN_MEMORY_HIGH;
    }
    else if(2 == memory) {
        model_info.memory = MNN_MEMORY_LOW;
    }
    else {
        model_info.memory = MNN_MEMORY_NORMAL;
    }
    return 0;
}

void InferEngine::setConfig(const EdModelInfo& model_info) {
    //set runtime config
    if (model_info.runtime == MNN_RUNTIME_AUTO) {
        config.type = MNN_FORWARD_AUTO;
    }
    else if (model_info.runtime == MNN_RUNTIME_CUDA) {
        config.type = MNN_FORWARD_CUDA;
    }
    else if (model_info.runtime == MNN_RUNTIME_OPENCL) {
        config.type = MNN_FORWARD_OPENCL;
    }
    else if (model_info.runtime == MNN_RUNTIME_VULKAN) {
        config.type = MNN_FORWARD_VULKAN;
    }
    else {
        config.type = MNN_FORWARD_CPU;
    }

    config.numThread = 1;

    //set Precision
    if (model_info.precision == MNN_PRECISION_NORMAL) {
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
    }
    else if (model_info.precision == MNN_PRECISION_HIGH) {
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_High;
    }
    else if (model_info.precision == MNN_PRECISION_LOW) {
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
    }

    //set Power
    if (model_info.power == MNN_POWER_NORMAL) {
        backendConfig.power = MNN::BackendConfig::PowerMode::Power_Normal;
    }
    else if(model_info.power == MNN_POWER_HIGH) {
        backendConfig.power = MNN::BackendConfig::PowerMode::Power_High;
    }
    else if(model_info.power == MNN_POWER_LOW) {
        backendConfig.power = MNN::BackendConfig::PowerMode::Power_Low;
    }

    //set Memory
    if (model_info.memory == MNN_MEMORY_NORMAL) {
        backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_Normal;
    }
    else if(model_info.memory == MNN_MEMORY_NORMAL) {
        backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_High;
    }
    else if(model_info.memory == MNN_MEMORY_NORMAL) {
        backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_Low;
    }

    config.backendConfig = &backendConfig;
}


InferEngine::InferEngine() {}

InferEngine::~InferEngine() {
    for (auto iter : inputDataHost) {
        inputDataHostItem = iter.second;
        delete inputDataHostItem;
    }

    for (auto iter : outputDataHost) {
        outputDataHostItem = iter.second;
        delete outputDataHostItem;
    }
}

int InferEngine::Init(EdModelInfo& model_info,
                    std::string model_path,
                    std::map<std::string, EdBuffer> input_data,
                    std::map<std::string, EdBuffer> output_data) {
                        
    net.reset(MNN::Interpreter::createFromFile((model_path+".mnn").c_str()));
    if (net == nullptr) {
        printf("MNN engine init failed! \n");
        return -1;
    }
    char *version = (char *)net->getModelVersion();
    printf("MNN version: %s \n", version);

    setConfig(model_info);

    session = net->createSession(config);
    inputData = net->getSessionInputAll(session);
    outputData = net->getSessionOutputAll(session);

    if (checkAndCreateInputBuffer(model_info.inputNode, input_data) != 0) return -1;
    if (checkAndCreateOutputBuffer(model_info.outputNode, output_data) != 0) return -1;

    return 0;
}

int InferEngine::checkAndCreateInputBuffer(const std::vector<EdNodeInfo> node_info,
                                        std::map<std::string, EdBuffer> buffer) {

                                            
    size_t input_nums = inputData.size();
    if (node_info.size() != input_nums) {
        printf("given net param Input nums %ld not match detect nums %ld ! \n", node_info.size(), input_nums);
        return -1;
    }
    if (buffer.size() != input_nums) {
        printf("given data Input nums %ld not match detect nums %ld ! \n", buffer.size(), input_nums);
        return -1;
    }

    std::vector<EdNodeInfo>::iterator in_node;
    for (auto node : node_info) {
        printf("given net input name: %s \n", node.name);
        Tensor *inputItem = inputData[node.name];
        printf("given net input size [%d %d %d %d] \n", inputItem->shape()[0], inputItem->shape()[1], inputItem->shape()[2], inputItem->shape()[3]);
    }

    std::map<std::string, Tensor*>::iterator iter;
    std::map<std::string, EdBuffer>::const_iterator buffer_iter;
    for (auto iter : inputData) {
        std::string inputName = iter.first;
        float *ptr = reinterpret_cast<float *>(buffer[inputName].bufferData);
        for (int j = 555; j < 565; j++) {
            printf("%f buffer \n", *(ptr + j));
        }
        Tensor *inputItem = iter.second;
        std::vector<int> inputShape = inputItem->shape();

        buffer_iter = buffer.find(std::string(inputName));
        if (buffer_iter == buffer.end()) {
            printf("data Input layer %s not match! \n", inputName.c_str());
            return -1;
        }

        EdBuffer currentBuffer = buffer[inputName];
        size_t buffer_size = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * \
                            float(inputItem->size()) / float(inputItem->elementSize());
        if (buffer_size != currentBuffer.bufferSize) {
            printf("input [%s] size check failed, net input: %ld, data buffer: %ld \n",
            inputName.c_str(), buffer_size, currentBuffer.bufferSize);
            return -1;
        }

        if (buffer[inputName].dataType != 32) {
            printf("Warning! MNN buffer may not support this datatype: %d\n", buffer[inputName].dataType);
        }
        std::vector<int> dims{1,  inputShape[2], inputShape[3], inputShape[1]};
        inputDataHostItem = MNN::Tensor::create<float>(dims, buffer[inputName].bufferData, MNN::Tensor::TENSORFLOW);

        std::pair<std::string, Tensor*> pair_item = std::make_pair(inputName, inputDataHostItem);
        inputDataHost.insert(pair_item);
    }
    return 0;
}

int InferEngine::checkAndCreateOutputBuffer(const std::vector<EdNodeInfo> node_info,
                                        std::map<std::string, EdBuffer> buffer) {
    size_t output_nums = outputData.size();
    if (node_info.size() != output_nums) {
        printf("given net param Output nums %ld not match detect nums %ld ! \n", node_info.size(), output_nums);
        return -1;
    }
    if (buffer.size() != output_nums) {
        printf("given data Output nums %ld not match detect nums %ld ! \n", buffer.size(), output_nums);
        return -1;
    }

    std::vector<EdNodeInfo>::iterator node;
    for (auto node : node_info) {
        printf("given net output name: %s \n", node.name);
        Tensor *outputItem = outputData[node.name];
        printf("given net output size [%d %d %d %d] \n", outputItem->shape()[0], outputItem->shape()[1], outputItem->shape()[2], outputItem->shape()[3]);
    }

    std::map<std::string, Tensor*>::iterator iter;
    std::map<std::string, EdBuffer>::const_iterator buffer_iter;
    for (auto iter : outputData) {
        std::string outputName = iter.first;
        Tensor *outputItem = iter.second;
        std::vector<int> outputShape = outputItem->shape();

        buffer_iter = buffer.find(std::string(outputName));
        if (buffer_iter == buffer.end()) {
            printf("data Output layer %s not match! \n", outputName.c_str());
            return -1;
        }

        EdBuffer currentBuffer = buffer[outputName];
        size_t buffer_size = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3] * \
                            float(outputItem->size()) / float(outputItem->elementSize());
        if (buffer_size != currentBuffer.bufferSize) {
            printf("output [%s] size check failed, net output: %ld, data buffer: %ld \n",
            outputName.c_str(), buffer_size, currentBuffer.bufferSize);
            return -1;
        }

        if (buffer[outputName].dataType != 32) {
            printf("Warning! MNN buffer may not support this datatype: %d\n", buffer[outputName].dataType);
        }

        std::vector<int> dims{1, outputShape[2], outputShape[3], outputShape[1]};
        outputDataHostItem = MNN::Tensor::create<float>(dims, buffer[outputName].bufferData, MNN::Tensor::TENSORFLOW);

        std::pair<std::string, Tensor*> pair_item = std::make_pair(outputName, outputDataHostItem);
        outputDataHost.insert(pair_item);
    }
    return 0;
}

int InferEngine::Inference() {
    int ret;
    std::map<std::string, Tensor*>::iterator iter;
    for (auto iter : inputData) {
        std::string name = iter.first;
        inputData[name]->copyFromHostTensor(inputDataHost[name]);
    }
    ret = net->runSession(session);
    if (ret != 0) {
        printf("ERROR: MNN run failed error code %d \n", ret);
        return -1;
    }
    for (auto iter : outputData) {
        std::string name = iter.first;
        outputData[name]->copyToHostTensor(outputDataHost[name]);
    }
    return 0;
}


