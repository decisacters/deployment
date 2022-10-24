
#ifndef NET_H
#define NET_H

#include "core/_util.h"
#include "core/yolov5.h"
#include "core/yolov7.h"
#include "core/mmpose.h"
#include "core/mmcls.h"

namespace ai
{
    class Net
    {
    public:
        Net(
            std::string model_path,
            unsigned int num_threads
        );
        ~Net();

        ai::YOLOV5 *yolov5;
        ai::YOLOV7 *yolov7;
        ai::MMPose *mmpose;
        ai::MMCLS *mmcls;

        std::map<int, std::vector<short>> color_map;
        std::string model_path;
 
        void predict(
            std::vector<ai::Mat<unsigned char>>& input_mats, 
            float conf_threshold, 
            Object& object
        );

        void predict(
            std::vector<ai::Mat<unsigned char>>& input_mats, 
            float conf_threshold, 
            float iou_threshold, 
            std::vector<Object>& object_list
        );

        void predict(
            std::vector<ai::Mat<unsigned char>>& input_mats, 
            float conf_threshold, 
            std::vector<std::vector<Point>>& point_list,
            std::vector<Object> object_list
        );
    };

} // namespace ai

#endif // NET_H