
#ifndef TRT_NET_H
#define TRT_NET_H

#include "core/_util.h"

#ifndef CUDA_FOUND

enum cudaError_t
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,
};

enum cudaStream_t;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

#endif

#ifndef TRT_FOUND

inline cudaError_t cudaMalloc(void **devPtr,size_t size) { return cudaSuccess; };
inline cudaError_t cudaMallocHost(void **ptr, size_t size) { return cudaSuccess; };
inline cudaError_t cudaFree(void *devPtr) { return cudaSuccess; };
inline cudaError_t cudaFreeHost(void *ptr) { return cudaSuccess; };
inline cudaError_t cudaStreamCreate(cudaStream_t *pStream) { return cudaSuccess; };
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) { return cudaSuccess; };
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return cudaSuccess; };
inline cudaError_t cudaSetDevice(int device) { return cudaSuccess; };
inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) { return cudaSuccess; };

namespace nvinfer1
{
    class ILogger
    {
        public:
        enum class Severity : int32_t
        {
            //! An internal error has occurred. Execution is unrecoverable.
            kINTERNAL_ERROR = 0,
            //! An application error has occurred.
            kERROR = 1,
            //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
            kWARNING = 2,
            //!  Informational messages with instructional information.
            kINFO = 3,
            //!  Verbose messages with debugging information.
            kVERBOSE = 4,
        };
        ILogger() {};
        virtual void log(Severity severity, char const* msg) noexcept = 0;
    };

    class Dims
    {
    public:
        Dims() {};
        //! The maximum number of dimensions supported for a tensor.
        static constexpr int32_t MAX_DIMS{8};
        //! The number of dimensions.
        int32_t nbDims;
        //! The extent of each dimension.
        int32_t d[MAX_DIMS];
    };

    class IExecutionContext
    {
        public:
        IExecutionContext() {};
        bool enqueue(int32_t batchSize, void* const* bindings, cudaStream_t stream, void* inputConsumed)
        { return true; };
    };

    class ICudaEngine
    {
        public:
        ICudaEngine() {};
        IExecutionContext* createExecutionContext() { return nullptr; };
        int32_t getNbBindings() { return 0; };
        bool bindingIsInput(int32_t bindingIndex) { return true; };
        const char* getBindingName(int32_t bindingIndex) { return nullptr; };
        Dims getBindingDimensions(int32_t bindingIndex) { Dims d; return d; };
        
    };

    class IRuntime
    {
        public:
        IRuntime() {};
        ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size) { return nullptr; };
    };

    inline IRuntime* createInferRuntime(ILogger& logger) { return nullptr; };
}

#else
#include "NvInfer.h"
#endif

namespace ai_trt
{
    class Logger : public nvinfer1::ILogger
    {
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override 
        {
            const char* serverity_info;
            switch (severity)
            {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: serverity_info = "[F] ";
                case nvinfer1::ILogger::Severity::kERROR: serverity_info = "[E] ";
                case nvinfer1::ILogger::Severity::kWARNING: serverity_info = "[W] ";
                case nvinfer1::ILogger::Severity::kINFO: serverity_info = "[I] ";
                case nvinfer1::ILogger::Severity::kVERBOSE: serverity_info = "[V] ";
                default: serverity_info = "";
            }
            printf("%s %s\n", serverity_info, msg);
        }

    };

    class TRTNet: public ai::NetUtil
    {
    protected:

        cudaError_t error_code;
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
        cudaStream_t stream;
        int32_t num_binding;
        uint8_t* input_host = nullptr;
        uint8_t* input_device = nullptr;

    public:
        explicit TRTNet(
            std::string model_path,
            unsigned int num_threads = 1
        );
        virtual ~TRTNet();

        std::vector<std::vector<ai::Mat<float>>> predict(std::vector<ai::Mat<unsigned char>> mats);

        char* deserialize(std::string trt_path, size_t& size);
        
        // TODO CUDA
        void preprocess_kernel_img(
            uint8_t* src, 
            int src_width, 
            int src_height,
            float* dst, 
            int dst_width, 
            int dst_height,
            cudaStream_t stream
        );

    };


} // namespace ai_trt

#endif // TRT_NET_H