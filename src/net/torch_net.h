
#ifndef TORCH_NET_H
#define TORCH_NET_H

#include "core/_util.h"

#ifndef TORCH_FOUND

namespace torch
{
    enum class DeviceType : int8_t 
    {
        CPU = 0,
        CUDA = 1, // CUDA.
        MKLDNN = 2, // Reserved for explicit MKLDNN
        OPENGL = 3, // OpenGL
        OPENCL = 4, // OpenCL
        IDEEP = 5, // IDEEP.
        HIP = 6, // AMD HIP
        FPGA = 7, // FPGA
        ORT = 8, // ONNX Runtime / Microsoft
        XLA = 9, // XLA / TPU
        Vulkan = 10, // Vulkan
        Metal = 11, // Metal
        XPU = 12, // XPU
        MPS = 13, // MPS
        Meta = 14, // Meta (tensors with no data)
        HPU = 15, // HPU / HABANA
        VE = 16, // SX-Aurora / NEC
        Lazy = 17, // Lazy Tensors
        IPU = 18, // Graphcore IPU
        PrivateUse1 = 19, // PrivateUse1 device
        // NB: If you add more devices:
        //  - Change the implementations of DeviceTypeName and isValidDeviceType
        //    in DeviceType.cpp
        //  - Change the number below
        COMPILE_TIME_MAX_DEVICE_TYPES = 20,
    };

    using DeviceIndex = int8_t;
    // DeviceType kCPU = DeviceType::CPU;
    class IValue;

    class Device
    {
    public:
        Device() {};
        Device(DeviceType type, DeviceIndex index = -1) {};
    };

    class Tensor
    {
    public:
        Tensor() {};
        Tensor to(Device device) { Tensor t; return t;};

        template <typename T>
        T * data_ptr() const;
    };

    namespace jit
    {

        class IValue
        {
        public:
            IValue() {};
            
            std::shared_ptr<IValue> toTuple() { std::shared_ptr<IValue> v; return v; };
            std::vector<IValue> elements() { std::vector<IValue> v; return v; };
            Tensor toTensor() { Tensor t; return t; };
            
            bool isTuple() { return false;};
            bool isTensorList() { return false;};
            bool isTensor() { return false;};
            IValue toList() { IValue t; return t; };
            IValue get(int output_index) { IValue t; return t; };
        };

        namespace script
        {
            class Module
            {
            public:
                Module() {};
                void eval() {};
                IValue forward(std::vector<IValue> inputs) { IValue v; return v; };
            };
        }
        
        using ExtraFilesMap = std::map<std::string, std::string>;

        inline script::Module load(
            const std::string& filename,
            Device device,
            ExtraFilesMap& extra_files)
        { script::Module m; return m; };

    }

    inline Tensor from_blob(
        void* data,
        std::vector<int64_t> sizes
    )
    { Tensor t; return t; };

}

namespace c10
{
    using Device = torch::Device;
}

#else
#include "torch/script.h"
#endif

namespace ai_torch
{
    class TorchNet: public ai::NetUtil
    {
    protected:
        torch::jit::script::Module module;
        torch::DeviceIndex device_index;
    
    public:
        explicit TorchNet(
            std::string model_path,
            unsigned int num_threads = 1
        );
        virtual ~TorchNet();

        std::vector<std::vector<ai::Mat<float>>> predict(std::vector<ai::Mat<unsigned char>> mats);

        torch::jit::IValue forward(
            std::vector<ai::Mat<unsigned char>> mats,
            std::vector<ai::Mat<float>>& input_mats
        );

    };


} // namespace ai_torch

#endif // TORCH_NET_H