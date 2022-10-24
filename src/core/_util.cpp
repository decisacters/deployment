
#include "core/_util.h"

#ifdef WIN32
std::string sep = "\\";
#else
std::string sep = "/";
#endif


ai::NetUtil::NetUtil(const std::string& model_path)
{
    std::string yaml_content = ai::read_yaml(model_path);

    if (yaml_content.empty()) return;
    
    std::vector<std::string> string_items;
    std::vector<float> float_items;
    
    ai::get_yaml_values(yaml_content, "norm_vals", string_items, norm_vals);
    ai::get_yaml_values(yaml_content, "mean_vals", string_items, mean_vals);

    if (norm_vals.empty())
        norm_vals = {1, 1, 1};
    if (mean_vals.empty())
        mean_vals = {0, 0, 0};
}


float ai::sigmoid(float x)
{
    #ifdef ANDROID
        return static_cast<float>(1.f / (1.f + exp(-x)));
    #else
        return static_cast<float>(1.f / (1.f + std::exp(-x)));
    #endif
}

short ai::rand_short(short max, short min)
{
    return min + (rand() % static_cast<short>(max - min + 1));
}

int ai::int2color(int* values)
{
    return values[0] * 65536 + values[1] * 256 + values[2];
}

int ai::shape2size(std::vector<int64_t> shape)
{
    int size = 1;
    for (size_t i = 0; i < shape.size(); i++)
        size *= (int)shape[i];
    return size;
}

std::string ai::shape2string(std::vector<int64_t> shape)
{
    std::string size = "";
    for (size_t i = 0; i < shape.size(); i++)
        size += " " + std::to_string(shape[i]);
    return size;
}

std::string ai::replace(std::string str_, const std::string& pattern, const std::string& replacement)
{
    std::string str = std::string(str_);
    std::vector<std::string> parts = ai::split(pattern, "|");
    for (auto & part : parts)
    {
        size_t pos;
        while ((pos = str.find(part)) != std::string::npos && pos < str.size())
            str.replace(pos, part.length(), replacement);
    }

    return str;
}

std::vector<std::string> ai::split(const std::string& s, const std::string& del)
{
    size_t start = 0;
    size_t end = s.find(del);
    std::vector<std::string> strings;
    while (end != -1)
    {
        strings.push_back(s.substr(start, end - start));
        start = end + del.size();
        end = s.find(del, start);
    }
    strings.push_back(s.substr(start, end - start));
    return strings;
}

std::string ai::read_text(const std::string& path)
{
    std::string content;
    std::ifstream file(path);

    if (file.is_open())
    {
        std::string line;
        while (getline(file, line))
            content += line + "\n";
        file.close();
    }
    else
        printf("Open %s failed\n", path.c_str());
    
    return content;
}

std::vector<std::string> ai::get_images(std::string img_dir)
{
    std::vector<std::string> images;

    if (img_dir.empty()) return images;

    #ifdef OPENCV_FOUND
    cv::glob(img_dir, images);
    #endif

    std::vector<std::string> filter_images;
    std::function<bool(std::string)> lambda = [](std::string s) { 
        return std::string(s).find(".jpg") != std::string::npos || 
                std::string(s).find(".JPEG") != std::string::npos; 
    };
    std::copy_if(images.begin(), images.end(), std::back_inserter(filter_images), lambda);
    
    return filter_images;
}

std::string ai::get_out_path(
    const std::string& file_path, 
    const std::string& out_dir, 
    const std::string& model_name
)
{
    // std::string pattern = *(ai::split(file_path, sep).rbegin()+1);
    // std::string out_path = ai::replace(file_path, filename, "out" + sep + model_name);
    std::string filename = *(ai::split(file_path, sep).rbegin());
    std::string out_path = out_dir + sep + model_name + sep + filename;
    return out_path;
}

std::string ai::get_model_name(const std::string& model_path, bool with_framework)
{
    std::string model_name = ai::split(model_path, sep).back();
    std::string framework = ai::split(ai::replace(model_path,  sep + model_name, ""), sep).back();
    std::string ext = "." + ai::split(model_path, ".").back();
    return ai::replace((with_framework ? framework : "") + sep + model_name, ext);
}

void ai::mkdirs(const std::string& path)
{
    std::vector<std::string> parts = ai::split(path, sep);
    std::string dir;
    struct stat buffer{};
    int result = -1;

    for (size_t i = 0; i < parts.size(); i++)
    {

        if (parts[i].empty()) continue;
        #ifdef WIN32
        dir += parts[i] + sep;
        #else
        dir += sep + parts[i];
        #endif

        if (stat(dir.c_str(), &buffer) != 0)
        {
            #ifdef _MSC_VER
            result = _wmkdir((const wchar_t*)dir.c_str());
            #else
            result = mkdir(dir.c_str(), 0777);
            #endif
            if (result == -1)
                printf("Could not create %s\n", dir.c_str());
        }
    }
}

int ai::get_device()
{
    size_t max_available_memory = 0;
    size_t available_memory = 0;
    int max_device_id = -1;
    int device_count = 0;

    #ifdef CUDA_FOUND
    // cudaGetDeviceCount(&device_count);
    // for (int device_id = 0; device_id < device_count; device_id++)
    // {
    //     cudaDeviceProp cuda_device_prop;
    //     cudaGetDeviceProperties(&cuda_device_prop, device_id);
    //     // TODO actual available memory
    //     available_memory = cuda_device_prop.totalGlobalMem;

    //     if (available_memory > max_available_memory && available_memory > 1.5 * 1073741824)
    //     {
    //         max_available_memory = available_memory;
    //         max_device_id = device_id;
    //     }
    // }
    #endif

    return max_device_id;
}

ai::Mat<unsigned char> ai::img2mat(unsigned char* data, int w, int h, int n, int c)
{
    ai::Mat<unsigned char> mat;
    mat.n = n;
    mat.c = c;
    mat.h = h;
    mat.w = w;
    std::vector<unsigned char> vec(data, data + mat.n * mat.c * mat.h * mat.w);
    mat.vec = vec;
    return mat;
}

void ai::to_mats(
    std::vector<ai::Mat<float>>& mats,
    std::map<std::string, ai::Mat<float>>& mat_map,
    std::string name,
    std::vector<int64_t> shape,
    float* data
)
{
    ai::Mat<float> mat;
    mat.n = (int)shape[0];
    mat.c = shape.size() < 2 ? 1 : (int)shape[1];
    mat.h = shape.size() < 3 ? 1 : (int)shape[2];
    mat.w = shape.size() < 4 ? 1 : (int)shape[3];
    mat.d = shape.size() < 5 ? 1 : (int)shape[4];
    std::vector<float> vec(data, data + mat.n * mat.c * mat.h * mat.w * mat.d);
    mat.vec = vec;
    mat_map[name] = mat;
    mats.push_back(mat);
}

void ai::make_parent_dirs(std::string out_path)
{
    std::string filename = *(ai::split(out_path, sep).rbegin());
    std::string out_dir = ai::replace(out_path, sep + filename);
    
    ai::mkdirs(out_dir);
}

void ai::save_image(ai::Mat<unsigned char> mat, std::string img_path, std::string out_dir, std::string model_name)
{
    std::string out_path = get_out_path(img_path, out_dir, model_name);
    ai::make_parent_dirs(out_path);

    bool result = false;
    #ifdef OPENCV_FOUND
    cv::Mat img(mat.h, mat.w, CV_8UC3, (unsigned char *)mat.vec.data());
    result = cv::imwrite(out_path, img);
    #endif

    if (result == false)
        printf("Could not save %s\n", out_path.c_str());
    else
        printf("Save image to %s\n", out_path.c_str());

}

ai::Mat<unsigned char> ai::load_image(std::string filename)
{
    ai::Mat<unsigned char> mat;
    #ifdef OPENCV_FOUND
    cv::Mat img = cv::imread(filename);
    mat = ai::img2mat(img.data, img.cols, img.rows);
    #endif

    if (mat.vec.empty())
        printf("Could not open %s\n", filename.c_str());
    else
        printf("Load image %s \n", filename.c_str());
    return mat;
}

std::map<std::string, ai::Mat<unsigned char>> ai::load_images(std::vector<std::string> filenames)
{

    std::map<std::string, ai::Mat<unsigned char>> mats;
    
    for (auto &&filename : filenames)
    {
        ai::Mat<unsigned char> mat = ai::load_image(filename);

        if (!mat.vec.empty())
            mats[filename] = mat;
    }
    return mats;
}


float* ai::set_input(
    const ai::Mat<unsigned char>& mat, 
    std::vector<int64_t> input_shape,
    std::vector<float> norm_vals,
    std::vector<float> mean_vals
)
{
    float *input_array = nullptr;
    #ifdef OPENCV_FOUND
    cv::Mat float_image;
    cv::Mat image_mat(mat.h, mat.w, CV_8UC3, (unsigned char *)mat.vec.data());
    cv::cvtColor(image_mat, float_image, cv::COLOR_BGR2RGB);
    cv::resize(float_image, float_image, cv::Size((int)input_shape[3], (int)input_shape[2]));
    
    float_image.convertTo(float_image, CV_32FC3);
    for (int i = 0; i < float_image.rows; ++i)
    {
        cv::Vec3f *p = float_image.ptr<cv::Vec3f>(i);
        for (int j = 0; j < float_image.cols; ++j)
        {
            p[j][0] = (p[j][0] / 255.f - mean_vals[0]) / norm_vals[0];
            p[j][1] = (p[j][1] / 255.f - mean_vals[1]) / norm_vals[1];
            p[j][2] = (p[j][2] / 255.f - mean_vals[2]) / norm_vals[2];
        }
    }

    input_array = new float[float_image.cols * float_image.rows * float_image.channels()];

    // hwc -> chw
    std::vector<cv::Mat> chw(float_image.channels());
    
    for (int i = 0; i < float_image.channels(); ++i)
        chw[i] = cv::Mat(cv::Size(float_image.cols, float_image.rows), CV_32FC1, input_array + i * float_image.cols * float_image.rows);
    
    cv::split(float_image, chw);
    // input_array = (float*)float_image.data;
    #endif
    return input_array;

}

void ai::get_yaml_values(
    const std::string& content, 
    const std::string& key,
    std::vector<std::string>& string_items, 
    std::vector<float>& float_items
)
{
    std::vector<float> values;
    size_t key_index = content.find(key);

    if(key_index == std::string::npos) return;

    size_t start_index = content.find('[', key_index);
    size_t end_index = content.find(']', start_index);
    std::string value = content.substr(start_index + 1, end_index - start_index - 1);

    string_items = ai::split(value, ",");

    for (std::string & string_item : string_items)
    {
        if (string_item.find(" #") != std::string::npos) continue;

        if (value.find('\"') != std::string::npos)
            string_item = ai::replace(string_item, " |\n|\r|\"");
        else
            float_items.push_back(std::stof(string_item));
    }
    
}


std::string ai::read_yaml(const std::string& model_path)
{
    std::string model_dir = ai::replace(model_path, sep + ai::split(model_path, sep).back(), "");
    std::string platform = ai::split(model_dir, sep).back();
    std::string model_name = ai::split(model_path, sep).back();
    std::string configs_dir = ai::replace(model_path, platform + sep + ai::split(model_path, sep).back(), "configs" + sep);
    
    size_t max_length = 0;
    std::string yaml_path, yaml_content;
    std::vector<std::string> configs;
    #ifdef OPENCV_FOUND
    cv::glob(configs_dir + sep + "*.yaml", configs);
    #endif
    for (size_t i = 0; i < configs.size(); i++)
    {
        std::string filename = ai::split(configs[i], sep).back();
        std::string ext = "." + ai::split(filename, ".").back();
        if (model_name.find(ai::replace(filename, ext)) != std::string::npos && filename.length() > max_length)
        {
            max_length = filename.length();
            yaml_path = configs_dir + sep + filename;
        }
    }

    if (max_length == 0)
        printf("Could not find yaml file for %s.\n", model_name.c_str());
    else
        yaml_content = ai::read_text(yaml_path);

    return yaml_content;
}
