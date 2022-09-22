#include "params.h"

void stringToInt(std::string s, std::vector<int>& nums) {
    int len = s.size();
    std::string tmp = "";
    for (int i = 0; i < len; i++){
        if(s[i] == ' ') {
            nums.push_back(atoi(tmp.c_str()));
            tmp = "";
        }
        else {
            tmp += s[i];
        }
    }
}

int getConfigs(InferConfig config, EdModelInfo& model_info) {
    printf("\n========================= start set params =========================\n");
    YAML::Node params;
    try {
        params = YAML::LoadFile(config.yaml_path);
    }
    catch (...) {
        printf("error loading file, yaml file %s error or not exist\n", config.yaml_path);
        return -1;
    }
    for (YAML::const_iterator iter = params.begin(); iter != params.end(); iter++) {
        std::string key = iter->first.as<std::string>();
        YAML::Node val = iter->second;
        switch (val.Type())
        {
        case YAML::NodeType::Scalar :
            printf("key: %s Scalar \n", key.c_str());
            break;
        case YAML::NodeType::Sequence:
            printf("key: %s Sequence \n", key.c_str());
            std::cout << "Seq: " << val <<std::endl;
            break;
        case YAML::NodeType::Map:
            printf("key: %s Map \n", key.c_str());
            break;
        case YAML::NodeType::Null:
            printf("key: %s NULL \n", key.c_str());
            break;
        case YAML::NodeType::Undefined:
            printf("key: %s Undefined \n", key.c_str());
            break;
        default:
            break;
        }
    }
    // set model_path
    std::string model_path = params["model_path"].as<std::string>();
    model_info.modelPath = model_path;

    for (auto io_name : params["model_io"]) {
        std::string io_names = io_name.first.as<std::string>();

        printf("=================== io name is: %s ==================\n", io_names.c_str());

        if (io_names == "input_info") {
            for (auto in_name : io_name.second) {
                EdNodeInfo input_node;
                std::string input_name = in_name.first.as<std::string>();
                //char *input_name = in_name.first.as<std::string>().c_str();

                input_node.name = const_cast<char *>(input_name.c_str());
                printf("input name: %s, ", in_name.first.as<std::string>().c_str());
                printf("shape and datatype: %s \n", in_name.second.as<std::string>().c_str());
                std::vector<int> in_res;
                stringToInt(in_name.second.as<std::string>().c_str(), in_res);

                input_node.shape[0] = in_res[0];
                input_node.shape[1] = in_res[1];
                input_node.shape[2] = in_res[2];
                input_node.shape[3] = in_res[3];
                input_node.dataType = in_res[4];
                model_info.inputNode.push_back(input_node);
            }
        }

        if (io_names == "output_info") {
            for (auto out_name : io_name.second) {
                EdNodeInfo output_node;
                std::string output_name = out_name.first.as<std::string>();
                output_node.name = const_cast<char *>(output_name.c_str());
                printf("output name: %s, ", output_name.c_str());
                printf("shape and datatype: %s \n", out_name.second.as<std::string>().c_str());
                std::vector<int> out_res;
                stringToInt(out_name.second.as<std::string>().c_str(), out_res);

                output_node.shape[0] = out_res[0];
                output_node.shape[1] = out_res[1];
                output_node.shape[2] = out_res[2];
                output_node.shape[3] = out_res[3];
                output_node.dataType = out_res[4];
                model_info.outputNode.push_back(output_node);
            }
        }
    }
    printf("===================== set params finished !!! =======================\n");
}
