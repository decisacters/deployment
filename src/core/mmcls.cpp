
#include "core/mmcls.h"

ai::MMCLS::MMCLS(
    std::string _model_path,
    std::map<int, std::vector<short>>& color_map_,
    unsigned int num_threads
):Framework(model_path, num_threads)
{
    read_yaml(model_path);
    color_map_ = color_map;
}


ai::MMCLS::~MMCLS()
{
}


void ai::MMCLS::read_yaml(const std::string& model_path)
{
    std::string yaml_content = ai::read_yaml(model_path);

    if (yaml_content.empty()) return;

    std::vector<float> float_items;
    std::vector<std::string> string_items;

    ai::get_yaml_values(yaml_content, "names", class_names, float_items);

    for (int i = 0; i < class_names.size(); i++)
        if (color_map.find(i) == color_map.end())
            color_map[i] = std::vector<short>{ai::rand_short(), ai::rand_short(), ai::rand_short()};

}

void ai::MMCLS::post_processing(
    ai::Mat<unsigned char> mat,
    float conf_threshold, 
    std::vector<std::vector<ai::Mat<float>>> vectors,
    Object& object
)
{
    ai::Mat<float> vector = vectors[1][0];
    size_t max_idx = std::max_element(vector.vec.begin(),vector.vec.end()) - vector.vec.begin();
    object.id = (int)max_idx;
    object.score = vector.vec[max_idx];
    object.name = (char*) class_names[max_idx].c_str();
}


std::vector<ai::Mat<unsigned char>> ai::MMCLS::pre_processing(ai::Mat<unsigned char>& mat)
{
    std::vector<ai::Mat<unsigned char>> input_mats;
    input_mats.push_back(mat);

    return input_mats;
}

void ai::MMCLS::predict(
    std::vector<ai::Mat<unsigned char>>& input_mats_, 
    float conf_threshold,
    Object& object
)
{
    std::vector<std::vector<ai::Mat<float>>> vectors;
    
    std::vector<ai::Mat<unsigned char>> input_mats = pre_processing(input_mats_[0]);

    // TODO stack batch
    for (size_t i = 0; i < input_mats.size(); i++)
    {
        vectors = Framework::predict(input_mats);

        if (!vectors.empty() && !vectors[0].empty() && !vectors[1].empty())
            post_processing(
                input_mats[0], 
                conf_threshold,
                vectors,
                object
            );
    }

}