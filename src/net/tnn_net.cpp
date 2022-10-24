
#ifdef TNN_FOUND

#include "net/tnn_net.h"

ai_tnn::Net::Net(std::string _proto_path,
                    std::string _model_path,
                    unsigned int num_threads,
                    TNN_NS::DeviceType _device_type
                    ): proto_path(_proto_path),
                        model_path(_model_path),
                        num_threads(num_threads),
                        device_type(_device_type)
{
    std::string proto_buffer, model_buffer;
    proto_buffer = ai_tnn::Net::load_buffer(proto_path);
    model_buffer = ai_tnn::Net::load_buffer(model_path);

    TNN_NS::ModelConfig model_config;
    // model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {proto_buffer, model_buffer};

    // 1. init TNN net
    TNN_NS::Status status;
    tnn_net = std::make_shared<TNN_NS::TNN>();
    status = tnn_net->Init(model_config);
    if (status != TNN_NS::TNN_OK || !tnn_net)
    {
    #ifdef TNN_DEBUG
        LOGE("net init failed %s", status.description().c_str());
    #endif
        return;
    }

    // 2. init instance
    TNN_NS::NetworkConfig network_config;
    // network_config.library_path = {""};
    network_config.device_type = device_type;

    instance = tnn_net->CreateInst(network_config, status);
    if (status != TNN_NS::TNN_OK || !instance)
    {
    #ifdef TNN_DEBUG
        LOGE("CreateInst failed%s", status.description().c_str());
    #endif
        return;
    }
    instance->SetCpuNumThreads((int) num_threads);

    // 3 input output map

    this->get_input_names();
    this->get_output_names();

}

ai_tnn::Net::~Net()
{
    tnn_net = nullptr;
    instance = nullptr;
}

std::string ai_tnn::Net::load_buffer(std::string path) 
{
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) 
    {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

void ai_tnn::Net::get_input_names()
{
    if (this->instance)
    {
        TNN_NS::BlobMap blob_map;
        this->instance->GetAllInputBlobs(blob_map);
        for (const auto &item : blob_map)
            this->input_names.push_back(item.first);
    }
}


void ai_tnn::Net::get_output_names()
{
    if (this->instance)
    {
        TNN_NS::BlobMap blob_map;
        this->instance->GetAllOutputBlobs(blob_map);
        for (const auto &item : blob_map)
            this->output_names.push_back(item.first);
    }
}


TNN_NS::DimsVector ai_tnn::Net::get_input_shape(std::string name) {
    TNN_NS::DimsVector dims_vector = {};
    TNN_NS::BlobMap blob_map = {};
    if (this->instance) {
        this->instance->GetAllInputBlobs(blob_map);
    }
    if (name == "" && blob_map.size() > 0) {
        if (blob_map.begin()->second) {
            dims_vector = blob_map.begin()->second->GetBlobDesc().dims;
        }
    }
    
    if (blob_map.find(name) != blob_map.end() && blob_map[name]) {
        dims_vector = blob_map[name]->GetBlobDesc().dims;
    }
    return dims_vector;
}

TNN_NS::DimsVector ai_tnn::Net::get_output_shape(std::string name) {
    TNN_NS::DimsVector dims_vector = {};
    TNN_NS::BlobMap blob_map = {};
    if (this->instance) {
        this->instance->GetAllOutputBlobs(blob_map);
    }
    if (name == "" && blob_map.size() > 0) {
        if (blob_map.begin()->second) {
            dims_vector = blob_map.begin()->second->GetBlobDesc().dims;
        }
    }
    
    if (blob_map.find(name) != blob_map.end() && blob_map[name]) {
        dims_vector = blob_map[name]->GetBlobDesc().dims;
    }
    return dims_vector;
}

TNN_NS::MatType ai_tnn::Net::get_input_mat_type(std::string name) {
    if (this->instance) {
        TNN_NS::BlobMap blob_map;
        this->instance->GetAllInputBlobs(blob_map);
        auto blob = (name == "") ? blob_map.begin()->second : blob_map[name];
        if (blob->GetBlobDesc().data_type == TNN_NS::DATA_TYPE_INT32) {
            return TNN_NS::NC_INT32;
        }
    }
    return TNN_NS::NCHW_FLOAT;
}

TNN_NS::MatType ai_tnn::Net::get_output_mat_type(std::string name) {
    if (this->instance) {
        TNN_NS::BlobMap blob_map;
        this->instance->GetAllOutputBlobs(blob_map);
        auto blob = (name == "") ? blob_map.begin()->second : blob_map[name];
        if (blob->GetBlobDesc().data_type == TNN_NS::DATA_TYPE_INT32) {
            return TNN_NS::NC_INT32;
        }
    }
    return TNN_NS::NCHW_FLOAT;
}

std::map<std::string, std::shared_ptr<TNN_NS::Mat> > ai_tnn::Net::get_input_mat_map()
{
    return this->input_mat_map;
}

std::map<std::string, std::shared_ptr<TNN_NS::Mat> > ai_tnn::Net::get_output_mat_map()
{
    return this->output_mat_map;
}

TNN_NS::Status ai_tnn::Net::Resize(std::shared_ptr<TNN_NS::Mat> src, 
                                    std::shared_ptr<TNN_NS::Mat> dst, 
                                    TNN_NS::InterpType interp_type) 
{
    
    TNN_NS::Status status = TNN_NS::TNN_OK;
    void * command_queue = nullptr;

    if (this->instance) {
        status = this->instance->GetCommandQueue(&command_queue);

        if (status != TNN_NS::TNN_OK) {
            LOGE("getCommandQueue failed with:%s\n", status.description().c_str());
            return status;
        }
    } else {
        return TNN_NS::Status(TNN_NS::TNNERR_INST_ERR, "instance_ GetCommandQueue return nil");
    }

    TNN_NS::ResizeParam param;
    param.type = interp_type;
    
    TNN_NS::DimsVector dst_dims = dst->GetDims();
    TNN_NS::DimsVector src_dims = src->GetDims();
    param.scale_w = dst_dims[3] / static_cast<float>(src_dims[3]);
    param.scale_h = dst_dims[2] / static_cast<float>(src_dims[2]);
    
    status = TNN_NS::MatUtils::Resize(*(src.get()), *(dst.get()), param, command_queue);
    if (status != TNN_NS::TNN_OK) {
        LOGE("resize failed with: %s\n", status.description().c_str());
    }
    
    return status;
}

void ai_tnn::Net::set_input_mat(ncnn::Mat img_bgr, int input_index)
{

    TNN_NS::Status status = TNN_NS::TNN_OK;
    TNN_NS::DimsVector image_shape, input_shape;
    std::shared_ptr<TNN_NS::Mat> image_mat, input_mat;
    
    image_shape = {1, 3, img_bgr.rows, img_bgr.cols};

    ncnn::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    image_mat = std::make_shared<TNN_NS::Mat>(this->device_type, 
                                                TNN_NS::N8UC3, 
                                                image_shape, 
                                                img_rgb.data);

    std::string input_name = this->input_names[input_index];

    input_shape = ai_tnn::Net::get_input_shape(input_name);
    
    if (input_shape.size() != 4)
    {
    #ifdef TNN_DEBUG
        LOGE("The input_shape only support 4 dims. Such as NCHW, NHWC ...");
    #endif
        return;
    }

    if ((image_shape[2] != input_shape[2] || image_shape[3] != input_shape[3])) 
    {
        input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), 
                                                image_mat->GetMatType(), 
                                                input_shape);
        status = this->Resize(image_mat, input_mat, TNN_NS::INTERP_TYPE_LINEAR);
    }
    else
        input_mat = image_mat;
    
    if (status != TNN_NS::TNN_OK) {
        LOGE("%s\n", status.description().c_str());
        return;
    }
    
    TNN_NS::MatConvertParam input_mat_convert_param = this->get_input_mat_convert_param();

    status = this->instance->SetInputMat(input_mat, input_mat_convert_param, input_name);

    if (status != TNN_NS::TNN_OK) {
        LOGE("instance SetInputMat Error: %s\n", status.description().c_str());
        return;
    }

    TNN_NS::MatConvertParam output_mat_convert_param = TNN_NS::MatConvertParam();
    this->input_mat_map[input_name] = input_mat;

}

void ai_tnn::Net::forward(std::vector<ncnn::Mat> input_mats)
{
    TNN_NS::Status status = TNN_NS::TNN_OK;

    for (size_t input_index = 0; input_index < input_mats.size(); input_index++)
        this->set_input_mat(input_mats[input_index], input_index);
    
    status = instance->Forward();
    if (status != TNN_NS::TNN_OK) {
        LOGE("instance Forward Error: %s\n", status.description().c_str());
        return;
    }

    for (std::string name : this->output_names) {
        TNN_NS::MatConvertParam output_mat_convert_param = TNN_NS::MatConvertParam();
        std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
        status = instance->GetOutputMat(output_mat, 
                                        output_mat_convert_param, 
                                        name,
                                        this->device_type,
                                        this->get_output_mat_type(name));
        this->output_mat_map[name] = output_mat;
    }
}

#endif
