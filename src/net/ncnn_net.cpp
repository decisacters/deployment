

#include "net/ncnn_net.h"

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int out_w = w / 2;
        int out_h = h / 2;
        int out_c = channels * 4;

        top_blob.create(out_w, out_h, out_c, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_c; p++)
        {
            #ifdef NCNN_FOUND
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            #else
            const float* ptr;
            #endif
            float* out_ptr = top_blob.channel(p);

            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    *out_ptr = *ptr;

                    out_ptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

ai_ncnn::NCNNNet::NCNNNet(
    const std::string& model_path,
    unsigned int num_threads
    ):NetUtil(model_path)
{
    ncnn_net = new ncnn::Net();

    std::string bin_path = model_path + ".bin";
    std::string param_path = model_path + ".param";
    
    #ifdef NCNN_FOUND
    if (bin_path.find("v.6.0") == std::string::npos && bin_path.find("yolov5") != std::string::npos)
        ncnn_net->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    #endif

    model_name = ai::get_model_name(model_path);

    printf("Loading NCNN model %s\n", model_name.c_str());
    
    int result;

    std::ifstream file(param_path);

    if (file.is_open())
        file.close();
    else
        printf("Open %s failed\n", param_path.c_str());

    result = ncnn_net->load_param(param_path.c_str());
    if (result != 0)
    {
        printf("net load_param %s failed", param_path.c_str());
        return;
    }

    result = ncnn_net->load_model(bin_path.c_str());
    if (result != 0)
    {
        printf("net load_bin %s failed", bin_path.c_str());
        return;
    }

    for (size_t i = 0; i < ncnn_net->input_names().size(); i++)
        input_names.push_back(ncnn_net->input_names()[i]);
    
    for (size_t i = 0; i < ncnn_net->output_names().size(); i++)
        output_names.push_back(ncnn_net->output_names()[i]);

}

ai_ncnn::NCNNNet::~NCNNNet()
{
    delete ncnn_net;
    ncnn_net = nullptr;
}


std::vector<std::vector<ai::Mat<float>>> ai_ncnn::NCNNNet::predict(std::vector<ai::Mat<unsigned char>> mats)
{

    ncnn::Extractor extractor = ncnn_net->create_extractor();

    ncnn::Mat input, output;

    // TODO put in parent class
    std::vector<ai::Mat<float>> input_mats, output_mats;
    vectors.clear();

    if (ncnn_net == nullptr) return vectors;

    for (int input_index = 0; input_index < input_names.size(); input_index++)
    {
        std::vector<int64_t> input_shape = input_shapes[input_index];

        float* input_array = ai::set_input(mats[input_index], input_shapes[input_index], norm_vals, mean_vals);

        input = ncnn::Mat((int)input_shape[2], (int)input_shape[3], (int)input_shape[1], input_array);

        std::string input_name = input_names[input_index];
        
        ai::to_mats(
            input_mats,
            input_mat_map,
            input_name,
            input_shapes[input_index],
            (float*)input.data
        );

        extractor.input(input_name.c_str(), input);
    }
    
    vectors.push_back(input_mats);

    for (int output_index = 0; output_index < output_names.size(); output_index++)
    {
        std::string output_name = output_names[output_index];
        extractor.extract(output_name.c_str(), output);
        
        std::vector<int64_t> output_shape = {output.d, output.c, output.w, output.h};

        ai::to_mats(
            output_mats,
            output_mat_map,
            output_name,
            output_shape,
            (float*)output.data
        );
    
    }

    vectors.push_back(output_mats);

    return vectors;

}
