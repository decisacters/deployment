
#ifndef MMCLS_H
#define MMCLS_H

#include "core/_util.h"
#include "core/_framework.h"

namespace ai
{
    class MMCLS: public Framework
    {

    public:
        MMCLS(
            std::string _model_path,
            std::map<int, std::vector<short>>& color_map,
            unsigned int num_threads
        );
        ~MMCLS();

        void read_yaml(const std::string& yaml_path);

        void post_processing(
            ai::Mat<unsigned char> mat,
            float conf_threshold, 
            std::vector<std::vector<ai::Mat<float>>> vectors,
            Object& object
        );

        std::vector<ai::Mat<unsigned char>> pre_processing(ai::Mat<unsigned char>& mat);

        void predict(
            std::vector<ai::Mat<unsigned char>>& input_mats, 
            float conf_threshold, 
            Object& object
        );
    };

} // namespace ai

#endif // MMCLS_H