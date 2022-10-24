
#include "core/mmpose.h"

ai::MMPose::MMPose(
    std::string _model_path,
    std::map<int, std::vector<short>>& color_map_,
    unsigned int num_threads
):Framework(model_path, num_threads)
{
    read_yaml(model_path);
    color_map_ = color_map;
}


ai::MMPose::~MMPose()
{
}


void ai::MMPose::read_yaml(const std::string& model_path)
{
    std::string yaml_content = ai::read_yaml(model_path);

    if (yaml_content.empty()) return;

    std::vector<std::string> string_items;
    std::vector<float> float_items;

    for (int i = 0; i < class_names.size(); i++)
        if (color_map.find(i) == color_map.end())
            color_map[i] = std::vector<short>{ai::rand_short(), ai::rand_short(), ai::rand_short()};

}

#if OPENCV_FOUND
// TODO cv::Point2f to ai::Point
cv::Point2f rotate_point(cv::Point2f pt, float angle_rad) 
{
    float sn = std::sin(angle_rad);
    float cs = std::cos(angle_rad);
    float new_x = pt.x * cs - pt.y * sn;
    float new_y = pt.x * sn + pt.y * cs;
    return {new_x, new_y};
}

cv::Point2f Get3rdPoint(cv::Point2f a, cv::Point2f b) 
{
    cv::Point2f direction = a - b;
    cv::Point2f third_pt = b + cv::Point2f(-direction.y, direction.x);
    return third_pt;
}

cv::Point2f operator*(cv::Point2f a, cv::Point2f b) 
{
    cv::Point2f c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
}

cv::Mat GetAffineTransform(
    cv::Point2f center, 
    cv::Point2f scale, 
    float rot,
    cv::Size output_size,
    cv::Point2f shift = {0.f, 0.f}, 
    bool inv = false)
{
    cv::Point2f scale_tmp = scale * 200;
    float src_w = scale_tmp.x;
    int dst_w = output_size.width;
    int dst_h = output_size.height;
    float rot_rad = 3.1415926f * rot / 180;
    cv::Point2f src_dir = rotate_point({0.f, src_w * -0.5f}, rot_rad);
    cv::Point2f dst_dir = {0.f, dst_w * -0.5f};

    cv::Point2f src_points[3];
    src_points[0] = center + scale_tmp * shift;
    src_points[1] = center + src_dir + scale_tmp * shift;
    src_points[2] = Get3rdPoint(src_points[0], src_points[1]);

    cv::Point2f dst_points[3];
    dst_points[0] = {dst_w * 0.5f, dst_h * 0.5f};
    dst_points[1] = dst_dir + cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);
    dst_points[2] = Get3rdPoint(dst_points[0], dst_points[1]);

    cv::Mat trans = inv ? cv::getAffineTransform(dst_points, src_points)
                        : cv::getAffineTransform(src_points, dst_points);
    return trans;
}
#endif

ai::Mat<float> get_max_pred(const ai::Mat<float>& heatmap) 
{

    int K = heatmap.c;
    int H = heatmap.h;
    int W = heatmap.w;

    int num_points = H * W;
    ai::Mat<float> pred;
    pred.c = K;
    pred.h = 3;
    pred.w = 1;
    pred.vec.resize(K * 3);

    for (int i = 0; i < K; i++) 
    {
        float* src_data = const_cast<float*>(heatmap.vec.data()) + i * H * W;
        double min_val = 0.0f, max_val = 0.0f;
        #if OPENCV_FOUND
        cv::Mat mat = cv::Mat(H, W, CV_32FC1, src_data);
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(mat, &min_val, &max_val, &min_loc, &max_loc);
        float* dst_data = pred.vec.data() + i * 3;
        *(dst_data + 0) = -1;
        *(dst_data + 1) = -1;
        *(dst_data + 2) = (float)max_val;
        if (max_val > 0.0) 
        {
            *(dst_data + 0) = (float)max_loc.x;
            *(dst_data + 1) = (float)max_loc.y;
        }
        #endif
    }

    return pred;
}

void transform_pred(
    ai::Mat<float>& pred, int k, const std::vector<float>& center, const std::vector<float>& _scale,
                      const std::vector<int>& output_size, bool use_udp = false) 
{
    auto scale = _scale;
    scale[0] *= 200;
    scale[1] *= 200;

    float scale_x, scale_y;
    if (use_udp) {
      scale_x = scale[0] / (output_size[0] - 1.0f);
      scale_y = scale[1] / (output_size[1] - 1.0f);
    } else {
      scale_x = scale[0] / output_size[0];
      scale_y = scale[1] / output_size[1];
    }

    float* data = pred.vec.data() + k * 3;
    *(data + 0) = *(data + 0) * scale_x + center[0] - scale[0] * 0.5f;
    *(data + 1) = *(data + 1) * scale_y + center[1] - scale[1] * 0.5f;
}

ai::Mat<float> keypoints_from_heatmap(
    const ai::Mat<float>& heatmap, 
    const std::vector<float>& center,
    const std::vector<float>& scale, 
    bool unbiased_decoding,        
    const std::string& post_process, 
    int modulate_kernel,
    float valid_radius_factor, 
    bool use_udp,
    const std::string& target_type)
{

    int K = heatmap.c;
    int H = heatmap.h;
    int W = heatmap.w;

    // if (post_process == "megvii") {
    //   heatmap = gaussian_blur(heatmap, modulate_kernel);
    // }

    ai::Mat<float> pred;

    if (use_udp) {
    //   if (to_lower(target_type) == to_lower(string("GaussianHeatMap"))) {
    //     pred = get_max_pred(heatmap);
    //     post_dark_udp(pred, heatmap, modulate_kernel);
    //   } else if (to_lower(target_type) == to_lower(string("CombinedTarget"))) {
    //     // output channel = 3 * channel_cfg['num_output_channels']
    //     assert(K % 3 == 0);
    //     cv::parallel_for_(cv::Range(0, K), _LoopBody{[&](const cv::Range& r) {
    //                         for (int i = r.start; i < r.end; i++) {
    //                           int kt = (i % 3 == 0) ? 2 * modulate_kernel + 1 : modulate_kernel;
    //                           float* data = heatmap.data<float>() + i * H * W;
    //                           cv::Mat work = cv::Mat(H, W, CV_32FC(1), data);
    //                           cv::GaussianBlur(work, work, {kt, kt}, 0);  // inplace
    //                         }
    //                       }});
    //     float valid_radius = valid_radius_factor_ * H;
    //     TensorDesc desc = {Device{"cpu"}, DataType::kFLOAT, {1, K / 3, H, W}};
    //     Tensor offset_x(desc);
    //     Tensor offset_y(desc);
    //     Tensor heatmap_(desc);
    //     {
    //       // split heatmap
    //       float* src = heatmap.data<float>();
    //       float* dst0 = heatmap_.data<float>();
    //       float* dst1 = offset_x.data<float>();
    //       float* dst2 = offset_y.data<float>();
    //       for (int i = 0; i < K / 3; i++) {
    //         std::copy_n(src, H * W, dst0);
    //         std::transform(src + H * W, src + 2 * H * W, dst1,
    //                        [=](float& x) { return x * valid_radius; });
    //         std::transform(src + 2 * H * W, src + 3 * H * W, dst2,
    //                        [=](float& x) { return x * valid_radius; });
    //         src += 3 * H * W;
    //         dst0 += H * W;
    //         dst1 += H * W;
    //         dst2 += H * W;
    //       }
    //     }
    //     pred = get_max_pred(heatmap_);
    //     for (int i = 0; i < K / 3; i++) {
    //       float* data = pred.data<float>() + i * 3;
    //       int index = *(data + 0) + *(data + 1) * W + H * W * i;
    //       float* offx = offset_x.data<float>() + index;
    //       float* offy = offset_y.data<float>() + index;
    //       *(data + 0) += *offx;
    //       *(data + 1) += *offy;
    //     }
    //   }
    }
    else 
    {
        pred = get_max_pred(heatmap);
    //   if (post_process == "unbiased") {
    //     heatmap = gaussian_blur(heatmap, modulate_kernel);
    //     float* data = heatmap.data<float>();
    //     std::for_each(data, data + K * H * W, [](float& v) {
    //       double _v = std::max((double)v, 1e-10);
    //       v = std::log(_v);
    //     });
    //     for (int i = 0; i < K; i++) {
    //       taylor(heatmap, pred, i);
    //     }

    //   } else if (post_process != "null") {
    //     for (int i = 0; i < K; i++) {
    //       float* data = heatmap.data<float>() + i * W * H;
    //       auto _data = [&](int y, int x) { return *(data + y * W + x); };
    //       int px = *(pred.data<float>() + i * 3 + 0);
    //       int py = *(pred.data<float>() + i * 3 + 1);
    //       if (1 < px && px < W - 1 && 1 < py && py < H - 1) {
    //         float v1 = _data(py, px + 1) - _data(py, px - 1);
    //         float v2 = _data(py + 1, px) - _data(py - 1, px);
    //         *(pred.data<float>() + i * 3 + 0) += (v1 > 0) ? 0.25 : ((v1 < 0) ? -0.25 : 0);
    //         *(pred.data<float>() + i * 3 + 1) += (v2 > 0) ? 0.25 : ((v2 < 0) ? -0.25 : 0);
    //         if (post_process_ == "megvii") {
    //           *(pred.data<float>() + i * 3 + 0) += 0.5;
    //           *(pred.data<float>() + i * 3 + 1) += 0.5;
    //         }
    //       }
    //     }
    //   }
    }

    K = pred.c;  // changed if target_type is CombinedTarget

    // Transform back to the image
    for (int i = 0; i < K; i++)
      transform_pred(pred, i, center, scale, std::vector<int>{W, H}, use_udp);

    // if (post_process_ == "megvii") {
    //   for (int i = 0; i < K; i++) {
    //     float* data = pred.data<float>() + i * 3 + 2;
    //     *data = *data / 255.0 + 0.5;
    //   }
    // }

    return pred;
}

void GetOutput(ai::Mat<float>& pred, std::vector<Point>& point_list) 
{
    int K = pred.c;
    float* data = pred.vec.data();
    for (int i = 0; i < K; i++) 
    {
        Point point;
        point.x = (int)*(data + 0);
        point.y = (int)*(data + 1);
        point.score = *(data + 2);
        point_list.push_back(point);
        data += 3;
    }
}

void box2cs(Object object, std::vector<float>& center, std::vector<float>& scale, float aspect_ratio)
{
    float x = (float)object.x1;
    float y = (float)object.y1;
    float w = (float)object.x2 - object.x1;
    float h = (float)object.y2 - object.y1;
    
    center.push_back(x + w * 0.5f);
    center.push_back(y + h * 0.5f);

    if (w > aspect_ratio * h)
        h = w * 1.0f / aspect_ratio;
    else if (w < aspect_ratio * h)
        w = h * aspect_ratio;

    scale.push_back(w / 200 * 1.25f);
    scale.push_back(h / 200 * 1.25f);
}

void ai::MMPose::post_processing(
    ai::Mat<unsigned char> mat,
    float conf_threshold, 
    std::vector<Point>& point_list,
    Object object,
    std::vector<std::vector<ai::Mat<float>>> vectors
)
{
    // mmdeploy/csrc/mmdeploy/codebase/mmpose/keypoints_from_heatmap.cpp

    ai::Mat<float> heatmap = vectors[1][0];
    std::vector<float> center, scale;
    box2cs(object, center, scale, (float)vectors[0][0].w / vectors[0][0].h);

    // TODO config
    // bool flip_test_{true};
    // bool shift_heatmap_{true};
    std::string post_process_ = {"default"};
    int modulate_kernel_{11};
    bool unbiased_decoding_{false};
    float valid_radius_factor_{0.0546875f};
    bool use_udp_{false};
    std::string target_type_{"GaussianHeatmap"};

    ai::Mat<float> pred = keypoints_from_heatmap(
        heatmap, center, scale, 
        unbiased_decoding_, post_process_,
        modulate_kernel_, valid_radius_factor_, 
        use_udp_, target_type_
    );
    
    GetOutput(pred, point_list);
}


std::vector<std::vector<ai::Mat<unsigned char>>> ai::MMPose::pre_processing(
    ai::Mat<unsigned char>& mat, 
    std::vector<Object> object_vector, 
    std::vector<int64_t> input_shape)
{
    // mmdeploy/csrc/mmdeploy/codebase/mmpose/topdown_affine.cpp
    std::vector<std::vector<ai::Mat<unsigned char>>> input_mats;
    std::vector<ai::Mat<unsigned char>> mats;

    for (size_t i = 0; i < object_vector.size(); i++)
    {
        mats.clear();
        Object object = object_vector[i];
        std::vector<float> center, scale;
        box2cs(object, center, scale, (float)input_shape[3] / input_shape[2]);
        
        #ifdef OPENCV_FOUND
        float rotation = 0.f;
        cv::Mat trans = GetAffineTransform(
            cv::Point2f{center[0], center[1]},
            cv::Point2f{scale[0], scale[1]},
            rotation,
            cv::Size{(int)input_shape[3], (int)input_shape[2]});
        
        cv::Mat src(mat.h, mat.w, CV_8UC3, mat.vec.data());
        cv::Mat dst;
        cv::warpAffine(src, dst, trans, cv::Size{(int)input_shape[3], (int)input_shape[2]});
        mats.push_back(ai::img2mat(dst.data, dst.cols, dst.rows));
        #endif

        input_mats.push_back(mats);
    }

    return input_mats;
}

void ai::MMPose::predict(
    std::vector<ai::Mat<unsigned char>>& input_mats_, 
    float conf_threshold, 
    std::vector<std::vector<Point>>& point_vector,
    std::vector<Object> object_vector
)
{
    std::vector<std::vector<ai::Mat<float>>> vectors;
    
    std::vector<std::vector<ai::Mat<unsigned char>>> input_mats = pre_processing(input_mats_[0], object_vector, input_shape);

    // TODO stack batch
    for (size_t i = 0; i < input_mats.size(); i++)
    {
        vectors = Framework::predict(input_mats[i]);

        std::vector<Point> point_vector_;

        if (!vectors.empty() && !vectors[0].empty() && !vectors[1].empty())
            post_processing(
                input_mats[i][0], 
                conf_threshold,
                point_vector_,
                object_vector[i],
                vectors
            );
        
        point_vector.push_back(point_vector_);
    }

}