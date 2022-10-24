
#ifndef TNN_NET_H
#define TNN_NET_H

#include "core/_net.h"

#include "tnn/core/tnn.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace ai_tnn
{
    class TNNNet: public ai::Net
    {
    protected:
        std::string proto_path;
        std::string model_path;
        std::shared_ptr<TNN_NS::TNN> tnn_net;
        std::shared_ptr<TNN_NS::Instance> instance;
        
        TNN_NS::DeviceType device_type;
        const unsigned int num_threads;
        std::map<std::string, std::shared_ptr<TNN_NS::Mat> > input_mat_map, output_mat_map;

        // int input_batch;
        // int input_channel;
        // int input_height;
        // int input_width;
        // int num_outputs = 1;
        // unsigned int input_value_size;
        // tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
        // tnn::MatType input_mat_type; // e.g NCHW_FLOAT
        // tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
        // tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
        // tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
        // // Actually, i prefer to hardcode the input/output names
        // // into subclasses, but we just let the auto detection here
        // // to make sure the debug information can show more details.
        // std::string input_name; // assume single input only.
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        // tnn::DimsVector input_shape; // vector<int>
        // std::map<std::string, tnn::DimsVector> output_shapes;
    private:
        /* data */
    public:
        explicit Net(std::string proto_path,
                    std::string model_path,
                    unsigned int num_threads = 1,
                    TNN_NS::DeviceType device_type = TNN_NS::DEVICE_NAIVE);
        virtual ~Net();
        
        static std::string load_buffer(std::string proto_or_model_path);

        void get_input_names();
        void get_output_names();

        TNN_NS::MatType get_input_mat_type(std::string name);
        TNN_NS::MatType get_output_mat_type(std::string name);
        
        TNN_NS::DimsVector get_input_shape(std::string name);
        TNN_NS::DimsVector get_output_shape(std::string name);

        std::map<std::string, std::shared_ptr<TNN_NS::Mat> > get_input_mat_map();
        std::map<std::string, std::shared_ptr<TNN_NS::Mat> > get_output_mat_map();

        void set_input_mat(ncnn::Mat mat, int input_index);
        void forward(std::vector<ncnn::Mat> input_mats);
        virtual TNN_NS::MatConvertParam get_input_mat_convert_param() = 0;
        TNN_NS::Status Resize(std::shared_ptr<TNN_NS::Mat> src, 
                                std::shared_ptr<TNN_NS::Mat> dst, 
                                TNN_NS::InterpType interp_type);

    };
} // namespace ai_tnn

#endif // TNN_NET_H