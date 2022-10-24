
#ifndef _NCNN_NET_H
#define _NCNN_NET_H

#include "core/_util.h"

#ifndef NCNN_FOUND
namespace ncnn
{
    
    class Allocator;
    class Mat
    {
    public:
        Mat() {};
        Mat(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0) {};
        int w, h, c, d;
        void* data;
        bool empty();
        float* channel(int c);
        const float* row(int y);
        void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    };
    class Extractor
    {
    public:
        int input(const char* blob_name, const Mat& in) { return 0; };
        int extract(const char* blob_name, Mat& feat, int type = 0) { return 0; };
    };
    class Option
    {
    public:
        Allocator *blob_allocator;
        int num_threads;
    };
    class Net
    {
    public:
        Net() {};
        Option opt;
        int load_param(const char* protopath) { return 0; };
        int load_model(const char* protopath) { return 0; };
        std::vector<const char*> input_names() 
        { std::vector<const char*> v; return v; };
        std::vector<const char*> output_names() 
        { std::vector<const char*> v; return v; };
        Extractor create_extractor() { Extractor e; return e; };
    };
    class Layer
    {
    public:
        bool one_blob_only;
        virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;
    };
    inline int get_big_cpu_count() { return 1; };
}
#define DEFINE_LAYER_CREATOR(name)
#else
#include "ncnn/net.h"
#include "ncnn/cpu.h"
#endif

namespace ai_ncnn
{
    class NCNNNet: public ai::NetUtil
    {

    public:
        
        ncnn::Net *ncnn_net;

        explicit NCNNNet(
            const std::string& model_path,
            unsigned int num_threads = 1
        );
        virtual ~NCNNNet();

        std::vector<std::vector<ai::Mat<float>>> predict(std::vector<ai::Mat<unsigned char>> mats);

    };

} // namespace ai_ncnn

#endif // _NCNN_NET_H