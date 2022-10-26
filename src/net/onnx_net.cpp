
#include "net/onnx_net.h"

ai_onnx::ONNXNet::ONNXNet(
    std::string model_path,
    unsigned int num_threads
    ):NetUtil(model_path)
{

    std::string onnx_path = model_path + ".onnx";

    ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, onnx_path.c_str());
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    
    int device_id = ai::get_device();

    #ifdef CUDA_FOUND
    OrtCUDAProviderOptions cuda_provider_options;
    cuda_provider_options.device_id = device_id;
    std::vector<std::string> available_providers = Ort::GetAvailableProviders();
    if (device_id >= 0 && std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end())
        session_options.AppendExecutionProvider_CUDA(cuda_provider_options);
    #endif

    model_name = ai::get_model_name(model_path);
    std::string device = device_id >= 0 ? "GPU" : "CPU";
    printf("Load ONNX model %s with Inference device: %s %d\n", model_name.c_str(), device.c_str(), device_id);

    #ifdef _MSC_VER
    ort_session = new Ort::Session(ort_env, (const wchar_t *)onnx_path.c_str(), session_options);
    #else
    ort_session = new Ort::Session(ort_env, onnx_path.c_str(), session_options);
    #endif

    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<ai::Mat<unsigned char>> mats;

    size_t num_inputs = ort_session->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++)
    {
        char* input_name = ort_session->GetInputName(i, allocator);
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        set_shapes(
            input_names,
            input_name,
            input_type_info,
            input_shapes
        );
        
        // warmup for GPU
        unsigned char* data = (unsigned char*)malloc(3 * input_shapes[0][2] * input_shapes[0][3] * sizeof(char));
        ai::Mat<unsigned char> mat = ai::img2mat(data, (int)input_shapes[0][2], (int)input_shapes[0][3]);
        mats.push_back(mat);
        printf("input name: %s, shape: %s\n", input_name, ai::shape2string(input_shapes[i]).c_str());
    }

    size_t num_outputs = ort_session->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i)
    {
        char* output_name = ort_session->GetOutputName(i, allocator);
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        set_shapes(
            output_names,
            output_name,
            output_type_info,
            output_shapes
        );
        printf("output name: %s, shape: %s\n", output_name, ai::shape2string(output_shapes[i]).c_str());
    }

    // inference once to warmup
    predict(mats);
}

ai_onnx::ONNXNet::~ONNXNet()
{
    delete ort_session;
    ort_session = nullptr;
}

std::vector<std::vector<ai::Mat<float>>> ai_onnx::ONNXNet::predict(std::vector<ai::Mat<unsigned char>> mats)
{
    std::vector<Ort::Value> input_values, output_values;

    // TODO put in parent class
    std::vector<ai::Mat<float>> input_mats, output_mats;
    vectors.clear();

    if (ort_session == nullptr) return vectors;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    for (int input_index = 0; input_index < input_names.size(); input_index++)
    {
        float* input_array = ai::set_input(mats[input_index], input_shapes[input_index], norm_vals, mean_vals);

        size_t input_shape_size = ai::shape2size(input_shapes[input_index]);;

        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_array,              
            input_shape_size, 
            input_shapes[input_index].data(),
            input_shapes[input_index].size()
        );

        std::string input_name = input_names[input_index];
        
        ai::to_mats(
            input_mats,
            input_mat_map,
            input_name,
            input_shapes[input_index],
            (float*)input.GetTensorData<float>()
        );

        input_values.push_back(std::move(input));

    }
    vectors.push_back(input_mats);

    char ** input_names_ = new char*[input_names.size()];
    for(size_t i = 0; i < input_names.size(); i++)
    {
        input_names_[i] = new char[input_names[i].size() + 1];
        #ifdef _MSC_VER
        strcpy_s(input_names_[i], input_names[i].size(), input_names[i].c_str());
        #else
        strcpy(input_names_[i], input_names[i].c_str());
        #endif
    }

    char ** output_names_ = new char*[output_names.size()];
    for(size_t i = 0; i < output_names.size(); i++)
    {
        output_names_[i] = new char[output_names[i].size() + 1];
        #ifdef _MSC_VER
        strcpy_s(output_names_[i], output_names[i].size(), output_names[i].c_str());
        #else
        strcpy(output_names_[i], output_names[i].c_str());
        #endif
    }
    
    output_values = ort_session->Run(
        Ort::RunOptions{nullptr},
        input_names_,
        input_values.data(),
        input_names.size(),
        output_names_,
        output_names.size()
    );
    
    for (int output_index = 0; output_index < output_names.size(); output_index++)
    {
        std::string output_name = output_names[output_index];

        ai::to_mats(
            output_mats,
            output_mat_map,
            output_name,
            output_shapes[output_index],
            (float*)output_values[output_index].GetTensorData<float>()
        );
    }
    vectors.push_back(output_mats);
    return vectors;

}

void ai_onnx::ONNXNet::set_shapes(
    std::vector<std::string>& names,
    const char* name,
    Ort::TypeInfo& type_info,
    std::vector<std::vector<int64_t>>& shapes
)
{
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();
    shapes.push_back(shape);
    names.push_back(name);
}
