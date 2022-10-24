
#include "net/torch_net.h"

void parse_shapes(
    std::string shapes_string, 
    std::string key, 
    std::vector<std::vector<int64_t>>& shapes,
    std::vector<std::string>& names
)
{
    std::string start_key = "[[";
    std::string end_key = "]]";
    size_t key_index = shapes_string.find(key);
    size_t start_index = shapes_string.find(start_key, key_index);
    size_t end_index = shapes_string.find(end_key, start_index);
    std::string value = shapes_string.substr(start_index + start_key.size(), end_index - start_index - end_key.size());
    
    std::vector<std::string> shapes_strings = ai::split(value, "], [");
    
    for (size_t i = 0; i < shapes_strings.size(); i++)
    {
        std::vector<std::string> dims_string = ai::split(shapes_strings[i], ",");
        std::vector<int64_t> dims;
        
        for (size_t j = 0; j < dims_string.size(); j++)
            dims.push_back(std::stoi(dims_string[j]));
        
        shapes.push_back(dims);
        std::string name = ai::replace(key, "_shapes") + std::to_string(i);
        names.push_back(name);
    }
}

ai_torch::TorchNet::TorchNet(
    std::string model_path,
    unsigned int num_threads
    ):NetUtil(model_path)
{
    std::string torch_path = model_path;

    device_index = ai::get_device();
    c10::Device device = torch::Device((int)device_index >= 0 ? torch::DeviceType::CUDA : torch::DeviceType::CPU, device_index);
    
    // https://github.com/ultralytics/yolov5/blob/master/export.py#L98-L99
    // d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    // extra_files = {'config.txt': json.dumps(d)}

    torch::jit::ExtraFilesMap extra_files_map;
    extra_files_map["shapes"] = ""; // string
    module = torch::jit::load(torch_path, device, extra_files_map); // TODO try catch
    module.eval();
    model_name = ai::get_model_name(model_path);
    printf("Load TORCH model %s with Inference device: %d \n", model_name.c_str(), device_index);

    if (extra_files_map["shapes"].find("output_shapes") == std::string::npos)
    {
        printf("extra_files_map do not have output_shapes\n");
        return;
    }
    
    
    if (extra_files_map["shapes"].find("input_shapes") == std::string::npos)
    {
        printf("extra_files_map do not have input_shapes\n");
        return;
    }

    parse_shapes(extra_files_map["shapes"], "output_shapes", output_shapes, output_names);
    parse_shapes(extra_files_map["shapes"], "input_shapes", input_shapes, input_names);
    
    for (size_t i = 0; i < 2; i++)
    {
        std::vector<ai::Mat<float>> input_mats;
        std::vector<ai::Mat<unsigned char>> mats;
        unsigned char *data = (unsigned char *)malloc(sizeof(unsigned char) * 3 * input_shapes[0][2] * input_shapes[0][3]);
        ai::Mat<unsigned char> mat = ai::img2mat(data, (int)input_shapes[0][2], (int)input_shapes[0][3]);
        mats.push_back(mat);
        torch::jit::IValue outputs = ai_torch::TorchNet::forward(mats, input_mats);
    }
    
}

ai_torch::TorchNet::~TorchNet()
{

}

torch::jit::IValue ai_torch::TorchNet::forward(
    std::vector<ai::Mat<unsigned char>> mats,
    std::vector<ai::Mat<float>>& input_mats
    )
{
    std::vector<torch::jit::IValue> inputs;
    c10::Device device = torch::Device((int)device_index >= 0 ? torch::DeviceType::CUDA : torch::DeviceType::CPU, device_index);

    for (int input_index = 0; input_index < input_names.size(); input_index++)
    {
        float* input_array = ai::set_input(mats[input_index], input_shapes[input_index], norm_vals, mean_vals);
        
        std::vector<int64_t> input_shape = input_shapes[input_index];

        torch::Tensor input = torch::from_blob(input_array, {input_shape[0], input_shape[1], input_shape[2], input_shape[3]}).to(device);

        std::string input_name = input_names[input_index];
        
        #ifdef TORCH_FOUND
        ai::to_mats(
            input_mats,
            input_mat_map,
            input_name,
            input_shapes[input_index],
            (float*)input.to(torch::kCPU).data_ptr<float>()
        );
        input.to(device);
        inputs.push_back(input);
        #endif
    }
    
    // torch::NoGradGuard no_grad_guard;
    torch::jit::IValue outputs = module.forward(inputs);

    return outputs;
}

std::vector<std::vector<ai::Mat<float>>> ai_torch::TorchNet::predict(std::vector<ai::Mat<unsigned char>> mats)
{

    // TODO put in parent class
    vectors.clear();
    std::vector<ai::Mat<float>> input_mats, output_mats;
    torch::jit::IValue outputs = ai_torch::TorchNet::forward(mats, input_mats);
    
    vectors.emplace_back(input_mats);

    for (int output_index = 0; output_index < output_names.size(); output_index++)
    {
        // TODO change 0 to output_index
        torch::Tensor output;
        if (outputs.isTuple())
            output = outputs.toTuple()->elements()[0].toTensor();
        else if (outputs.isTensorList())
            output = outputs.toList().get(output_index).toTensor();
        else if (outputs.isTensor())
            output = outputs.toTensor();

        std::string output_name = output_names[output_index];
        #ifdef TORCH_FOUND
        ai::to_mats(
            output_mats,
            output_mat_map,
            output_name,
            output_shapes[output_index],
            (float*)output.to(torch::kCPU).data_ptr<float>()
        );
        #endif
    }
    vectors.push_back(output_mats);
    return vectors;

}
