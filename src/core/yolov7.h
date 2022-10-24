
#ifndef YOLOV7_H
#define YOLOV7_H

#include "core/_util.h"
#include "core/_framework.h"

namespace ai
{
    struct YoloV7Anchor
    {
        int grid0;
        int grid1;
        int stride;
        float width;
        float height;
    };

    class YOLOV7: public Framework
    {

    public:
        YOLOV7(
            std::string _model_path,
            std::map<int, std::vector<short>>& color_map,
            unsigned int num_threads
        );
        ~YOLOV7();

        void predict(
            std::vector<ai::Mat<unsigned char>>& input_mats, 
            float conf_threshold, 
            float iou_threshold, 
            std::vector<Object>& object_list
        );

        std::vector<float> strides, anchor_grids;
        std::map<size_t, std::vector<YoloV7Anchor>> center_anchors;
        
        void read_yaml(const std::string& yaml_path);

        static void nms(
            std::vector<Object> &input, 
            std::vector<Object> &output, 
            float iou_threshold
        );
        
        void generate_anchors(int in_shape);

        void post_processing(
            const ai::Mat<unsigned char>& mat,
            float conf_threshold, 
            float iou_threshold,
            std::vector<Object>& object_list,
            std::vector<std::vector<ai::Mat<float>>> vectors
        );

        void pre_processing(ai::Mat<unsigned char>& mat);
    };

} // namespace ai

#endif // YOLOV7_H