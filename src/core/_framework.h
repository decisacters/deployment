
#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include "core/_util.h"
#include "headers.h"

#include "net/ncnn_net.h"

#include "net/onnx_net.h"

#include "net/torch_net.h"

#include "net/trt_net.h"

#ifdef OVINO_FOUND
#include "net/ovino_net.h"
#endif

namespace ai
{
    class Framework
    {
    public:
        Framework(
            std::string model_path,
            unsigned int num_threads
        );
        ~Framework();

        std::vector<std::vector<ai::Mat<float>>> predict(
            std::vector<ai::Mat<unsigned char>> input_mats);

        std::vector<std::string> class_names;
        std::map<int, std::vector<short>> color_map;
        std::vector<int64_t> input_shape;
        std::string model_path;

        ai_ncnn::NCNNNet *ncnn_net;

        ai_onnx::ONNXNet *onnx_net;
        
        ai_torch::TorchNet *torch_net;
        
        ai_trt::TRTNet *trt_net;

        #ifdef OVINO_FOUND
        ai_ovino::OVINONet *ovino_net;
        #endif
    };

} // namespace ai

#endif // FRAMEWORK_H
