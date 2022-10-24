#include "core/_framework.h"


ai::Framework::Framework(
    std::string _model_path,
    unsigned int num_threads
)
{
    model_path = _model_path;

    if (model_path.find("ncnn") != std::string::npos)
        ncnn_net = new ai_ncnn::NCNNNet(model_path, num_threads);

    if (model_path.find("onnx") != std::string::npos)
        onnx_net = new ai_onnx::ONNXNet(model_path, num_threads);
    
    if (model_path.find("torch") != std::string::npos)
        torch_net = new ai_torch::TorchNet(model_path, num_threads);

    if (model_path.find("trt") != std::string::npos)
        trt_net = new ai_trt::TRTNet(model_path, num_threads);

    #ifdef OVINO_FOUND
    if (model_path.find("ovino") != std::string::npos)
        ovino_net = new ai_ovino::OVINONet(model_path, num_threads);
    #endif
}


ai::Framework::~Framework()
{
    if (model_path.find("ncnn") != std::string::npos)
        delete ncnn_net;

    if (model_path.find("onnx") != std::string::npos)
        delete onnx_net;
    
    if (model_path.find("torch") != std::string::npos)
        delete torch_net;

    if (model_path.find("trt") != std::string::npos)
        delete trt_net;

    #ifdef OVINO_FOUND
    if (model_path.find("ovino") != std::string::npos)
        delete ovino_net;
    #endif
    
}

std::vector<std::vector<ai::Mat<float>>> ai::Framework::predict(
    std::vector<ai::Mat<unsigned char>> input_mats)
{
    std::vector<std::vector<ai::Mat<float>>> vectors;
    
    if (model_path.find("ncnn") != std::string::npos)
    {
        if (ncnn_net->input_shapes.empty())
            ncnn_net->input_shapes.push_back(input_shape);
        vectors = ncnn_net->predict(input_mats);
    }

    if (model_path.find("onnx") != std::string::npos)
        vectors = onnx_net->predict(input_mats);
    
    if (model_path.find("torch") != std::string::npos)
        vectors = torch_net->predict(input_mats);

    if (model_path.find("trt") != std::string::npos)
        vectors = trt_net->predict(input_mats);

    #ifdef OVINO_FOUND
    if (model_path.find("ovino") != std::string::npos)
        vectors = ovino_net->predict(input_mats);
    #endif

    return vectors;
}
