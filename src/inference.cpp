#include "inference.h"

inference::inference() {}

inference::~inference() {
    // if (infer_engine) {
    //     delete infer_engine;
    //     infer_engine = nullptr;
    // }
    // printf("East Deploy release success! \n");
}

int inference::Init(InferConfig config, InferModelConfig model_config) {
    infer_engine = new InferEngine();
    if (nullptr == infer_engine) {
        printf("Infer Engine create failed ! \n");
        return -1;
    }
    EdModelInfo model_info = {};
    model_info.modelPath = "seg";
    setRuntime(model_config.runtime, model_info);
    setPrecision(model_config.precision, model_info);
    setPower(model_config.power, model_info);
    setMemory(model_config.memory, model_info);

    EdNodeInfo input_node;
    input_node.dataType = 32;
    input_node.name = "input";
    input_node.shape[0] = 1;
    input_node.shape[1] = 768;
    input_node.shape[2] = 768;
    input_node.shape[3] = 3;

    EdNodeInfo output_node;
    output_node.dataType = 32;
    output_node.name = "output";
    output_node.shape[0] = 1;
    output_node.shape[1] = 768;
    output_node.shape[2] = 768;
    output_node.shape[3] = 1;

    model_info.inputNode.push_back(input_node);
    model_info.outputNode.push_back(output_node);

    std::string model_path = config.assets_path + std::string("/") + std::string(model_info.modelPath);

    int flag;
    flag = infer_engine->Init(model_info, model_path, model_config.input, model_config.output);
    if (flag !=0 ){
        printf("Infer Engine init failed ! \n");
        return -1;
    }
    printf("Infer Engine init success ! \n");

    return 0;
}

int inference::process() {
    int flag;
    flag = infer_engine->Inference();
    if (flag) {
        printf("Infer Engine process failed ! \n");
        return -1;
    }
    printf("Infer Engine process success ! \n");

    printf("\n======================= show result =======================\n");
    return 0;
}