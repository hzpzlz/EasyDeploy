#include "inference.h"
#include "params.h"

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
    setRuntime(model_config.runtime, model_info);
    setPrecision(model_config.precision, model_info);
    setPower(model_config.power, model_info);
    setMemory(model_config.memory, model_info);

    getConfigs(config, model_info);

    int flag;
    flag = infer_engine->Init(model_info, model_info.modelPath, model_config.input, model_config.output);
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