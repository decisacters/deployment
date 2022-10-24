
#ifndef NET_UTIL_H
#define NET_UTIL_H

#include "headers.h"
#include "daai.h"

#ifdef OPENCV_FOUND
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

#ifdef CUDA_FOUND
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace ai
{
    template<typename T> struct Mat 
    {
        int n, c, h, w, d;
        std::vector<T> vec;
    };

    class NetUtil
    {
    public:
        NetUtil(
            const std::string& model_path
        );
        std::string model_name, model_path, yaml_content;
        unsigned int num_threads;
        std::vector<std::string> input_names, output_names;
        std::vector<std::vector<int64_t>> input_shapes, output_shapes;
        std::map<std::string, ai::Mat<float>> input_mat_map, output_mat_map;
        std::vector<float> norm_vals, mean_vals;
        std::vector<std::vector<ai::Mat<float>>> vectors;
        std::vector<ai::Mat<float>> input_mats, output_mats;
    };

    int get_device();

    ai::Mat<unsigned char> img2mat(unsigned char* data, int w, int h, int n = 1, int c = 3);

    void to_mats(
        std::vector<ai::Mat<float>>& mats,
        std::map<std::string, ai::Mat<float>>& mat_map,
        std::string name,
        std::vector<int64_t> shape,
        float* data
    );

    ai::Mat<unsigned char> load_image(std::string filename);
    std::map<std::string, ai::Mat<unsigned char>> load_images(std::vector<std::string> image_filenames);
    void save_image(ai::Mat<unsigned char> mat, std::string img_path, std::string out_dir, std::string model_name);
    void draw_objects(unsigned char* pixels, int w, int h, 
        Object* objects, int max_count, std::map<int, std::vector<short>> color_map);
    float* set_input(
        const ai::Mat<unsigned char>& mat, 
        std::vector<int64_t> input_shape,
        std::vector<float> norm_vals,
        std::vector<float> mean_vals
    );
    
    float sigmoid(float x);
    short rand_short(short max = 255, short min = 0);
    int int2color(int* values);
    int shape2size(std::vector<int64_t> shape);
    std::string shape2string(std::vector<int64_t> shape);

    std::string get_model_name(const std::string& model_path, bool with_framework = false);
    std::string get_out_path(
        const std::string& img_path, 
        const std::string& out_dir, 
        const std::string& model_name
    );
    std::vector<std::string> get_images(std::string img_dir);
    void make_parent_dirs(std::string out_path);
    void mkdirs(const std::string& path);
    
    std::vector<std::string> split(const std::string& s, const std::string& del = " ");
    std::string replace(std::string str, const std::string& pattern, const std::string& replacement = "");
    std::string read_text(const std::string& path);

    void get_yaml_values(
        const std::string& content, 
        const std::string& key,
        std::vector<std::string>& string_items, 
        std::vector<float>& float_items
    );
    std::string read_yaml(const std::string& model_path);

} // namespace ai

#endif // NET_UTIL_H