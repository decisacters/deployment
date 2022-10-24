
#include "core/_net.h"

ai::Net::Net(
    std::string _model_path,
    unsigned int num_threads
)
{
    model_path = _model_path;

    if (std::string(model_path).find("yolov5") != std::string::npos)
        yolov5 = new ai::YOLOV5(model_path, color_map, num_threads);

    if (std::string(model_path).find("yolov7") != std::string::npos)
        yolov7 = new ai::YOLOV7(model_path, color_map, num_threads);

    if (std::string(model_path).find("mmpose") != std::string::npos)
        mmpose = new ai::MMPose(model_path, color_map, num_threads);

    if (std::string(model_path).find("mmcls") != std::string::npos)
        mmcls = new ai::MMCLS(model_path, color_map, num_threads);
}


ai::Net::~Net()
{
    if (std::string(model_path).find("yolov5") != std::string::npos)
    {
        delete yolov5;
        yolov5 = nullptr;
    }
    if (std::string(model_path).find("yolov7") != std::string::npos)
    {
        delete yolov7;
        yolov7 = nullptr;
    }
    if (std::string(model_path).find("mmpose") != std::string::npos)
    {
        delete mmpose;
        mmpose = nullptr;
    }
    if (std::string(model_path).find("mmcls") != std::string::npos)
    {
        delete mmcls;
        mmcls = nullptr;
    }
}

void ai::Net::predict(
    std::vector<ai::Mat<unsigned char>>& input_mats, 
    float conf_threshold, 
    Object& object
)
{
    if (model_path.find("mmcls") != std::string::npos)
        mmcls->predict(input_mats, conf_threshold, object);
}

void ai::Net::predict(
    std::vector<ai::Mat<unsigned char>>& input_mats, 
    float conf_threshold, 
    float iou_threshold, 
    std::vector<Object>& object_vector
)
{
    if (model_path.find("yolov5") != std::string::npos)
        yolov5->predict(input_mats, conf_threshold, iou_threshold, object_vector);
    if (model_path.find("yolov7") != std::string::npos)
        yolov7->predict(input_mats, conf_threshold, iou_threshold, object_vector);
}

void ai::Net::predict(
    std::vector<ai::Mat<unsigned char>>& input_mats, 
    float conf_threshold, 
    std::vector<std::vector<Point>>& point_vector,
    std::vector<Object> object_vector
)
{
    if (model_path.find("mmpose") != std::string::npos)
        mmpose->predict(input_mats, conf_threshold, point_vector, object_vector);
}
