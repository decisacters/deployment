
#ifndef ONNX_NET_H
#define ONNX_NET_H

#include "core/_util.h"

#ifndef ONNX_FOUND
enum OrtLoggingLevel 
{
    ORT_LOGGING_LEVEL_VERBOSE,  ///< Verbose informational messages (least severe).
    ORT_LOGGING_LEVEL_INFO,     ///< Informational messages.
    ORT_LOGGING_LEVEL_WARNING,  ///< Warning messages.
    ORT_LOGGING_LEVEL_ERROR,    ///< Error messages.
    ORT_LOGGING_LEVEL_FATAL,    ///< Fatal error messages (most severe).
};

enum OrtAllocatorType 
{
  OrtInvalidAllocator = -1,
  OrtDeviceAllocator = 0,
  OrtArenaAllocator = 1
};

enum OrtMemType 
{
    OrtMemTypeCPUInput = -2,              ///< Any CPU memory used by non-CPU execution provider
    OrtMemTypeCPUOutput = -1,             ///< CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    OrtMemTypeCPU = OrtMemTypeCPUOutput,  ///< Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
    OrtMemTypeDefault = 0,                ///< The default allocator for execution provider
};

class OrtCUDAProviderOptions
{
public:
    int device_id;
};

namespace Ort
{
    inline std::vector<std::string> GetAvailableProviders()
    { std::vector<std::string> p; return p; };

    class TensorTypeAndShapeInfo;

    class RunOptions
    {
    public:
        RunOptions(const char* run_tag) {};
    };
        
    template <typename T>
    class Unowned
    {
    public:
        Unowned() {};
        std::vector<int64_t> GetShape() { std::vector<int64_t> s; return s; };
    };

    
    class AllocatorWithDefaultOptions
    {
    public:
        AllocatorWithDefaultOptions() {};
    };

    class Env
    {
    public:
        Env(OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, const char* logid = "") {};
    };
  
    class TypeInfo
    {
    public:
        Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const
        { Unowned<TensorTypeAndShapeInfo> u; return u;};
    };

    class MemoryInfo
    {
    public:
        static MemoryInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type1)
        { MemoryInfo m; return m; };
    };

    class Value
    {
    public:
        template <typename T>
        static Value CreateTensor(const MemoryInfo info, T* p_data, 
            size_t p_data_element_count, const int64_t* shape, size_t shape_len)
        { Value v; return v; };
        template <typename T>
        const T* GetTensorData() const { return nullptr; };
    };

    class SessionOptions
    {
    public:
        SessionOptions SetIntraOpNumThreads(int intra_op_num_threads)
        { SessionOptions s; return s; };
        SessionOptions AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options)
        { SessionOptions s; return s; };
    };

    class Session
    {
    public:
    #ifdef _MSC_VER
        Session(Env& env, const wchar_t * model_path, const SessionOptions& options) {};
    #else
        Session(Env& env, const char * model_path, const SessionOptions& options) {};
    #endif
        size_t GetInputCount() const { return 0; };
        size_t GetOutputCount() const { return 0; };

        char* GetInputName(size_t index, AllocatorWithDefaultOptions allocator) const
        { return nullptr; };
        char* GetOutputName(size_t index, AllocatorWithDefaultOptions allocator) const
        { return nullptr; };

        TypeInfo GetInputTypeInfo(size_t index) const
        { TypeInfo t; return t; };

        TypeInfo GetOutputTypeInfo(size_t index) const
        { TypeInfo t; return t; };

        std::vector<Value> Run(
            const RunOptions& run_options, const char* const* input_names, 
            const Value* input_values, size_t input_count,
            const char* const* output_names, size_t output_count)
        { std::vector<Value> v; return v; };
    };

}

#else
#include "onnxruntime_cxx_api.h"
#endif

namespace ai_onnx
{
    class ONNXNet: public ai::NetUtil
    {
    protected:

        Ort::Env ort_env;
        Ort::Session *ort_session = nullptr;

    public:
        explicit ONNXNet(
            std::string model_path,
            unsigned int num_threads = 1
        );
        virtual ~ONNXNet();

        std::vector<std::vector<ai::Mat<float>>> predict(std::vector<ai::Mat<unsigned char>> mats);

        void set_shapes(
            std::vector<std::string>& names,
            const char* name,
            Ort::TypeInfo& type_info,
            std::vector<std::vector<int64_t>>& shapes
        );

    };


} // namespace ai_onnx

#endif // ONNX_NET_H