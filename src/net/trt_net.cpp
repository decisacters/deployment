

#include "net/trt_net.h"

ai_trt::TRTNet::TRTNet(
    std::string model_path,
    unsigned int num_threads
    ):NetUtil(model_path)
{

    std::string trt_path = model_path + ".engine";    
    int device_id = 0; // ai::get_device();

    model_name = ai::get_model_name(model_path);
    std::string device = device_id >= 0 ? "GPU" : "CPU";
    printf("Load TRT model %s with Inference device: %s %d \n", model_name.c_str(), device.c_str(), device_id);

    cudaSetDevice(device_id);

    ai_trt::Logger logger;
    runtime = nvinfer1::createInferRuntime(logger);
    if(runtime == nullptr)
    {
        printf("CreateInferRuntime failed\nline:%d func:%s file:%s\n", __LINE__, __func__, __FILE__);
        return;
    }

    // deserialize the .engine and run inference
    size_t size = 0;
    char *trt_model_stream = deserialize(trt_path, size);
    engine = runtime->deserializeCudaEngine(trt_model_stream, size);
    if(engine == nullptr)
    {
        printf("DeserializeCudaEngine failed\nline:%d func:%s file:%s\n", __LINE__, __func__, __FILE__);
        return;
    }

    context = engine->createExecutionContext();
    if(context == nullptr)
    {
        printf("CreateExecutionContext failed\nline:%d func:%s file:%s\n", __LINE__, __func__, __FILE__);
        return;
    }

    num_binding = engine->getNbBindings();
    std::vector<ai::Mat<unsigned char>> mats;

    for (int32_t binding_index = 0; binding_index < num_binding; binding_index++)
    {
        bool is_input = engine->bindingIsInput(binding_index);
        const char* name = engine->getBindingName(binding_index);
        nvinfer1::Dims dims = engine->getBindingDimensions(binding_index);
        std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);

        if (is_input)
        {
            input_names.push_back(name);
            input_shapes.push_back(shape);

            // warmup for GPU
            unsigned char* data = (unsigned char*)malloc(3 * input_shapes[0][2] * input_shapes[0][3] * sizeof(unsigned char));
            ai::Mat<unsigned char> mat = ai::img2mat(data, (int)input_shapes[0][2], (int)input_shapes[0][3]);
            mats.push_back(mat);
        }
        else
        {
            output_names.push_back(name);
            output_shapes.push_back(shape);
        }
        
    }

    error_code = cudaStreamCreate(&stream);
    if (error_code != cudaSuccess) 
        printf("cudaStreamCreate failed with error code %d\n", error_code);

    int max_input_size = 3000;
    // prepare input data cache in pinned memory 
    error_code = cudaMallocHost((void**)&input_host, max_input_size*max_input_size*3);
    if (error_code != cudaSuccess) 
        printf("cudaMallocHost failed with error code %d\n", error_code);
    // prepare input data cache in device memory
    error_code = cudaMalloc((void**)&input_device, max_input_size*max_input_size*3);
    if (error_code != cudaSuccess) 
        printf("cudaMalloc failed with error code %d\n", error_code);

    // inference once to warmup
    predict(mats);

}

ai_trt::TRTNet::~TRTNet()
{

    // Destroy the engine
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();

    error_code = cudaFree(input_device);
    error_code = cudaFreeHost(input_host);

    // Release stream and buffers
    cudaStreamDestroy(stream);
}

char* ai_trt::TRTNet::deserialize(std::string trt_path, size_t& size)
{
    char *trt_model_stream = nullptr;
    std::ifstream file(trt_path, std::ios::binary);
    if (!file.good()) 
    {
        printf("Open file %s failed", trt_path.c_str());
        return trt_model_stream;
    }
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    file.read(trt_model_stream, size);
    file.close();
    return trt_model_stream;

}

std::vector<std::vector<ai::Mat<float>>> ai_trt::TRTNet::predict(std::vector<ai::Mat<unsigned char>> mats)
{
    
    // TODO put in parent class
    std::vector<ai::Mat<float>> input_mats, output_mats;
    vectors.clear();

    float** buffers = (float**)malloc(num_binding * sizeof(float));

    for (int input_index = 0; input_index < input_names.size(); input_index++)
    {
        std::string input_name = input_names[input_index];

        std::vector<int64_t> input_shape = input_shapes[input_index];

        size_t input_shape_size = ai::shape2size(input_shape);

        float* input_array = (float*)malloc(input_shape_size * sizeof(float));

        error_code = cudaMalloc((void**)&buffers[input_index], input_shape_size * sizeof(float));
        if (error_code != cudaSuccess) 
        {
            printf("cudaMalloc input (name: %s, shape: %s) failed with error code %d\n", 
                input_name.c_str(), ai::shape2string(input_shape).c_str(), error_code
            );
            return vectors;
        }

        float* buffer_idx = (float*)buffers[input_index];
        
        // TODO remove yolov5 letterbox
        #if OPENCV_FOUND
        cv::Mat img(mats[input_index].h, mats[input_index].w, 
            CV_8UC3, (unsigned char *)mats[input_index].vec.data());

        cv::resize(img, img, cv::Size((int)input_shape[3], (int)input_shape[2]));
        mats[input_index] = ai::img2mat(img.data, img.cols, img.rows);
        #endif
        
        size_t size_image = mats[input_index].c * mats[input_index].h * mats[input_index].w;
        memcpy(input_host, mats[input_index].vec.data(), size_image);

        //copy data to device memory
        error_code = cudaMemcpyAsync(input_device, input_host, size_image, cudaMemcpyHostToDevice, stream);
        
        if (error_code != cudaSuccess) 
            printf("cudaMemcpyAsync failed with error code %d\n", error_code);
        
        #ifdef TRT_FOUND
        preprocess_kernel_img(input_device, (int)mats[input_index].w, (int)mats[input_index].h, 
            buffer_idx, input_shape[2], input_shape[3], stream);
        #endif
        
        ai::to_mats(
            input_mats,
            input_mat_map,
            input_name,
            input_shapes[input_index],
            input_array
        );

    }
    vectors.push_back(input_mats);

    for (int output_index = 0; output_index < output_names.size(); output_index++)
    {

        std::vector<int64_t> output_shape = output_shapes[output_index];

        std::string output_name = output_names[output_index];

        size_t output_shape_size = ai::shape2size(output_shape);

        float* output_array = (float*)malloc(output_shape_size * sizeof(float));

        error_code = cudaMalloc((void**)&buffers[input_names.size() + output_index], output_shape_size * sizeof(float));
        if (error_code != cudaSuccess) 
        {
            printf("cudaMalloc output (name: %s, shape: %s) failed with error code %d\n", 
                output_name.c_str(), ai::shape2string(output_shape).c_str(), error_code
            );
            return vectors;
        }

        bool result = context->enqueue((int)output_shape[0], (void **)buffers, stream, nullptr);
        if (!result)
            printf("context.enqueue infer failed\n");

        error_code = cudaMemcpyAsync(output_array, buffers[input_names.size() + output_index], 
            output_shape_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        
        if (error_code != cudaSuccess) 
        {
            printf("cudaMemcpyAsync %s (name: %s, shape: %s) failed with error code %d\n", 
                    false ? "input" : "output", output_name.c_str(),
                    ai::shape2string(output_shape).c_str(), error_code
            );
            return vectors;
        }
        
        cudaStreamSynchronize(stream);

        ai::to_mats(
            output_mats,
            output_mat_map,
            output_name,
            output_shapes[output_index],
            output_array
        );
    }
    vectors.push_back(output_mats);

    return vectors;

}
