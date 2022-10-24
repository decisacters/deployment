
#include "core/yolov7.h"

ai::YOLOV7::YOLOV7(
    std::string model_path,
    std::map<int, std::vector<short>>& color_map_,
    unsigned int num_threads
):Framework(model_path, num_threads)
{
    read_yaml(model_path);
    color_map_ = color_map;
}


ai::YOLOV7::~YOLOV7()
{
}


void ai::YOLOV7::nms(std::vector<Object> &input, std::vector<Object> &output, float iou_threshold) 
{

    std::sort(input.begin(), input.end(), [](const Object &a, const Object &b) { return a.score > b.score; });
    output.clear();

    size_t box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (size_t i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<Object> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        int h0 = input[i].y2 - input[i].y1 + 1;
        int w0 = input[i].x2 - input[i].x1 + 1;

        int area0 = h0 * w0;

        for (size_t j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            int inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            int inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            int inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            int inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            int inner_h = inner_y1 - inner_y0 + 1;
            int inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            int inner_area = inner_h * inner_w;

            int h1 = input[j].y2 - input[j].y1 + 1;
            int w1 = input[j].x2 - input[j].x1 + 1;

            int area1 = h1 * w1;

            float score;

            score = (float)inner_area / (float)(area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);

    }
}

void ai::YOLOV7::generate_anchors(int in_shape)
{
    for (size_t i = 0; i < strides.size(); i++)
    {
        int stride = (int) strides[i];
        unsigned int num_grid_w = in_shape / stride;
        unsigned int num_grid_h = in_shape / stride;
        std::vector<ai::YoloV7Anchor> anchors;
        size_t anchor_size = anchor_grids.size() / strides.size() / 2;
        for (size_t j = 0; j < anchor_size; j++)
        {
            for (size_t g1 = 0; g1 < num_grid_h; ++g1)
            {
                for (size_t g0 = 0; g0 < num_grid_w; ++g0)
                {
                    ai::YoloV7Anchor anchor;
                    anchor.grid0 = (int)g0;
                    anchor.grid1 = (int)g1;
                    anchor.stride = stride;

                    anchor.width = anchor_grids[i * anchor_size * 2 + j * 2];
                    anchor.height = anchor_grids[i * anchor_size * 2 + j * 2 + 1];

                    anchors.push_back(anchor);
                }
            }
        }
        center_anchors[i] = anchors;
    }

}

void ai::YOLOV7::read_yaml(const std::string& model_path)
{
    std::string yaml_content = ai::read_yaml(model_path);

    if (yaml_content.empty()) return;

    std::vector<float> float_items;
    std::vector<std::string> string_items;
    
    ai::get_yaml_values(yaml_content, "names", class_names, float_items);
    ai::get_yaml_values(yaml_content, "strides", string_items, strides);
    ai::get_yaml_values(yaml_content, "anchor_grids", string_items, anchor_grids);
    ai::get_yaml_values(yaml_content, "input_shapes", string_items, float_items);

    int in_shape = !float_items.empty() ? (int)float_items[0] : 640;
    Framework::input_shape = {1,3,in_shape,in_shape};
    generate_anchors(in_shape);

    for (int i = 0; i < class_names.size(); i++)
        if (color_map.find(i) == color_map.end())
            color_map[i] = std::vector<short>{ai::rand_short(), ai::rand_short(), ai::rand_short()};

}

void ai::YOLOV7::post_processing(
    const ai::Mat<unsigned char>& mat,
    float conf_threshold, 
    float iou_threshold,
    std::vector<Object>& object_list,
    std::vector<std::vector<ai::Mat<float>>> vectors
)
{

    std::vector<Object> extracted_objects;
    
    object_list.clear();
    unsigned int count = 0;

    int in_w = vectors[0][0].w;
    int in_h = vectors[0][0].h;

    float rw = (float)in_w / (float)mat.w;
    float rh = (float)in_h / (float)mat.h;

    bool with_output = (vectors[1][0].c / vectors[1][0].h) > 99;

    for (size_t n = 0; n < vectors[1].size(); n++)
    {
        if (with_output && n > 0) continue;

        ai::Mat<float> output = vectors[1][n];
        float *data = output.vec.data();
        int stride = (int)strides[n];

        // have c=3 indicate 3 anchors at one grid
        std::vector<ai::YoloV7Anchor> &stride_anchors = center_anchors[n];

        // e.g, 3*80*80 + 3*40*40 + 3*20*20 = 25200
        const unsigned int num_anchors = with_output ? output.c : output.c * in_w * in_h / stride / stride;
        const unsigned int num_classes = output.d > 5 ? output.d - 5: output.h - 5; // 80

        for (unsigned int i = 0; i < num_anchors; ++i)
        {
            float obj_conf, cls_conf;
            const float *offset_obj_cls_ptr;
            if (!with_output)
                offset_obj_cls_ptr = (float *) data + (i * (num_classes + 5)); // row ptr;

            obj_conf = with_output ? data[i*(num_classes + 5) + 4] : ai::sigmoid(offset_obj_cls_ptr[4]);
            cls_conf = with_output ? data[i*(num_classes + 5) + 5] : ai::sigmoid(offset_obj_cls_ptr[5]);
            
            if (obj_conf < conf_threshold) continue; // filter first.

            int label = 0;
            for (size_t j = 0; j < num_classes; ++j)
            {
                float tmp_conf = with_output ? data[i*(num_classes + 5) + j + 5] : ai::sigmoid(offset_obj_cls_ptr[j + 5]);
                if (tmp_conf > cls_conf)
                {
                    cls_conf = tmp_conf;
                    label = (int)j;
                }
            } // argmax

            float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
            if (conf < conf_threshold) continue; // filter

            float cx,cy,w,h,dx,dy,dw,dh;
            if (with_output)
            {
                dw = 0;
                dh = 0;
                cx = data[i*(num_classes + 5) + 0];
                cy = data[i*(num_classes + 5) + 1];
                w = data[i*(num_classes + 5) + 2];
                h = data[i*(num_classes + 5) + 3];
            }
            else
            {
                int grid0 = stride_anchors.at(i).grid0; // w
                int grid1 = stride_anchors.at(i).grid1; // h
                float anchor_w = stride_anchors.at(i).width;
                float anchor_h = stride_anchors.at(i).height;
                
                dx = ai::sigmoid(offset_obj_cls_ptr[0]);
                dy = ai::sigmoid(offset_obj_cls_ptr[1]);
                dw = ai::sigmoid(offset_obj_cls_ptr[2]);
                dh = ai::sigmoid(offset_obj_cls_ptr[3]);

                cx = (dx * 2.f - 0.5f + (float) grid0) * (float) stride;
                cy = (dy * 2.f - 0.5f + (float) grid1) * (float) stride;
                #ifdef ANDROID
                w = (float) pow(dw * 2.f, 2) * anchor_w;
                h = (float) pow(dh * 2.f, 2) * anchor_h;
                #else
                w = (float)std::pow(dw * 2.f, 2) * anchor_w;
                h = (float)std::pow(dh * 2.f, 2) * anchor_h;
                #endif
            }
            float x1 = ((cx - w / 2.f) - (float) dw) / rw;
            float y1 = ((cy - h / 2.f) - (float) dh) / rh;
            float x2 = ((cx + w / 2.f) - (float) dw) / rw;
            float y2 = ((cy + h / 2.f) - (float) dh) / rh;

            Object box{};
            box.x1 = std::max(0, (int)x1);
            box.y1 = std::max(0, (int)y1);
            box.x2 = std::min((int)x2, 9999);
            box.y2 = std::min((int)y2, 9999);
            box.score = conf;
            box.id = label;
            box.name = (char*)class_names[label].c_str();
            extracted_objects.push_back(box);

            count += 1; // limit boxes for nms.
        }
    }

    nms(extracted_objects, object_list, iou_threshold);
}

void ai::YOLOV7::pre_processing(ai::Mat<unsigned char>& mat)
{
    // TODO letterbox
}

void ai::YOLOV7::predict(
    std::vector<ai::Mat<unsigned char>>& input_mats, 
    float conf_threshold, 
    float iou_threshold, 
    std::vector<Object>& object_list
    )
{
    std::vector<std::vector<ai::Mat<float>>> vectors;
    pre_processing(input_mats[0]);
    
    vectors = Framework::predict(input_mats);

    if (!vectors.empty() && !vectors[0].empty() && !vectors[1].empty())
        post_processing(
            input_mats[0], 
            conf_threshold, 
            iou_threshold,
            object_list,
            vectors
        );
}