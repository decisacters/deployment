
#ifndef MMPOSE_H
#define MMPOSE_H

#include "core/_util.h"
#include "core/_framework.h"

namespace ai
{
    class MMPose: public Framework
    {

    public:
        MMPose(
            std::string _model_path,
            std::map<int, std::vector<short>>& color_map,
            unsigned int num_threads
        );
        ~MMPose();

        void read_yaml(const std::string& yaml_path);

        void post_processing(
            ai::Mat<unsigned char> mat,
            float conf_threshold, 
            std::vector<Point>& point_list,
            Object object,
            std::vector<std::vector<ai::Mat<float>>> vectors
        );

        std::vector<std::vector<ai::Mat<unsigned char>>> pre_processing(
            ai::Mat<unsigned char>& mat, 
            std::vector<Object> object_list,
            std::vector<int64_t> input_shape
        );

        void predict(
            std::vector<ai::Mat<unsigned char>>& input_mats, 
            float conf_threshold, 
            std::vector<std::vector<Point>>& point_list,
            std::vector<Object> object_list
        );
    };

} // namespace ai

#endif // MMPOSE_H