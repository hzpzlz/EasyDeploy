#include "ncnn_class.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "post_process.h"

int setRuntime(int runtime, EdModelInfo& model_info) {

    return 0;
}

int setPrecision(int precision, EdModelInfo& model_info) {

    return 0;
}

int setPower(int power, EdModelInfo& model_info) {

    return 0;
}

int setMemory(int memory, EdModelInfo& model_info) {

    return 0;
}

InferEngine::InferEngine() {}

InferEngine::~InferEngine() {

}

void InferEngine::setConfig(const EdModelInfo& model_info) {

    return;
}

int InferEngine::Init(EdModelInfo& model_info,
                    std::string model_path,
                    std::map<std::string, EdBuffer> input_data,
                    std::map<std::string, EdBuffer> output_data) {

    net.opt.use_vulkan_compute = true;
    std::string path_param = model_path + ".param";
    std::string path_bin = model_path+ ".bin";

    net.load_param(path_param.c_str());
    net.load_model(path_bin.c_str());

    setConfig(model_info);

    if (checkIOBuffer(model_info.inputNode, input_data, model_info.outputNode, output_data) != 0) return -1;

    return 0;
}

int InferEngine::checkIOBuffer(const std::vector<EdNodeInfo> in_node_info, std::map<std::string, EdBuffer> in_buffer,
                    const std::vector<EdNodeInfo> out_node_info, std::map<std::string, EdBuffer> out_buffer) {
    ncnn::Extractor ex = net.create_extractor();

    for (auto iter : in_node_info) {
        const char *inputName = iter.name;
        printf("intput info-> name: %s buffer_size: %ld \n", inputName, in_buffer[inputName].bufferSize);
        // unsigned char *ptr = reinterpret_cast<unsigned char *>(in_buffer[inputName].bufferData);
        // for (int j = 555; j < 565; j++) {
        //     printf("%d input buffer \n", *(ptr + j));
        // }
        
        ncnn::Mat inputData = ncnn::Mat::from_pixels_resize((unsigned char*)in_buffer[inputName].bufferData, 2, iter.shape[1], iter.shape[2], 227, 227);
        const float mean_values[3] = {0.f, 0.f, 0.f};
        const float val_values[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        inputData.substract_mean_normalize(mean_values, val_values);

        ex.input(inputName, inputData);
    }

    for (auto iter : out_node_info) {
        const char *outputName = iter.name;
        printf("output info-> name: %s buffer_size: %ld \n", outputName, out_buffer[outputName].bufferSize);
        ncnn::Mat outMat;
        ex.extract(outputName, outMat);
        // printf("%d %d %d %d\n", outMat.dims, outMat.h, outMat.w, outMat.c);

        // float *ptr = reinterpret_cast<float *>(outMat.data);
        // for (int j = 0; j < 10; j++) {
        //     printf("%f out buffer \n", *(ptr + j));
        // }

        if (out_buffer[outputName].dataType == 32) {
            ::memcpy(out_buffer[outputName].bufferData, outMat.data, sizeof(float) * 1000);
        }
        else if (out_buffer[outputName].dataType == 8) {
            ::memcpy(out_buffer[outputName].bufferData, outMat.data, sizeof(unsigned char) * 1000);
        }
    }

    return 0;
}

int InferEngine::Inference() {
    int ret = 0;
    if (ret != 0) {
        printf("ERROR: MNN run failed error code %d \n", ret);
    }
    return 0;
}


