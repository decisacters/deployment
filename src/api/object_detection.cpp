
#include "api/object_detection.h"

int save_objects(const char* out_path_, Object* objects, int object_count)
{
    std::string out_path = std::string(out_path_);
    std::string ext = *(ai::split(out_path, ".").rbegin());
    out_path = ai::replace(out_path, ext, "csv");
    std::ofstream outfile;
    outfile.open(out_path);
    if (outfile.is_open())
    {
        printf("Save %d object(s) to %s\n", (int)object_count, out_path.c_str());
        outfile << "x1,y1,x2,y2,score,id,name\n";
        for (size_t i = 0; i < object_count; i++)
        {
            Object object = objects[i];
            outfile << object.x1 << ","
                    << object.y1 << ","
                    << object.x2 << ","
                    << object.y2 << ","
                    << object.score << ","
                    << object.id << ","
                    << object.name << "\n";
        }

    }
    else
        printf("Could not open %s to save\n", out_path.c_str());
    
    return 0;
}

void draw_objects(unsigned char* pixels, int w, int h, std::map<int, std::vector<short>> color_map, Object* objects, int object_count)
{
    #ifdef OPENCV_FOUND
    cv::Mat mat(h, w, CV_8UC3, pixels);
    cv::Scalar color;
    for (size_t i = 0; i < object_count; i++)
    {
        Object box = objects[i];

        color = cv::Scalar{(double)color_map[box.id][0],(double)color_map[box.id][1],(double)color_map[box.id][2]};

        float font_scale = (float)((box.x2 - box.x1) * (box.y2 - box.y1)) / (float)w / (float)h;
        font_scale = font_scale < 0.5 ? 0.5f : font_scale;
        int font_pixel_size = (int) (font_scale * 20);
        int thickness = font_pixel_size / 5 + 1;
        int baseline=0;

        cv::rectangle(
            mat, 
            cv::Point(box.x1, box.y1), 
            cv::Point(box.x2, box.y2), 
            color,
            thickness
        );

        std::string text = std::string(box.name) + " " + std::to_string(box.score).substr(2, 2);

        int font_face = cv::FONT_HERSHEY_SIMPLEX;

        cv::Size text_size = cv::getTextSize(
            text.c_str(), 
            font_face, 
            font_scale, 
            thickness, 
            &baseline
        );

        cv::rectangle(
            mat, 
            cv::Point(box.x1, box.y1),
            cv::Point(box.x1+text_size.width, box.y1+text_size.height),
            color, 
            -1
        );

        cv::putText(
            mat, 
            text, 
            cv::Point(box.x1, box.y1+text_size.height),
            font_face,
            font_scale, 
            cv::Scalar(255, 255, 255)
        );
    }
    #endif
}

int detect(
    unsigned char* pixels, 
    int image_width, 
    int image_height,
    float conf_threshold, 
    Object* objects, 
    int& object_count,
    float iou_threshold
)
{
    std::vector<ai::Mat<unsigned char>> input_mats;
    input_mats.push_back(ai::img2mat(pixels, image_width, image_height));

    std::vector<Object> object_list;

    ai::Net* net = get_net();

    net->predict(input_mats, conf_threshold, iou_threshold, object_list);

    object_count = (int) object_list.size();

    std::copy(object_list.begin(), object_list.end(), objects);

    draw_objects(pixels, input_mats[0].w, input_mats[0].h, net->color_map, objects, object_count);

    return 0;
}

int test_object_detection(
    const char* model_path, 
    const char* in_dir, 
    const char* out_dir, 
    float conf_threshold, 
    int& object_count, 
    float iou_threshold
)
{
    printf("conf %f, iou %f\n", conf_threshold, iou_threshold);

    if (in_dir == nullptr) return -1;

    std::vector<std::string> filenames = ai::get_images(in_dir);

    if (!filenames.empty() && init(model_path, 1) != 0) return -1;

    for (std::string filename : filenames)
    {
        ai::Mat<unsigned char> image = ai::load_image(filename);

        if (image.vec.empty()) continue;

        std::string model_name = ai::get_model_name(model_path, true);
        
        clock_t start, end;
        start = clock();

        Object objects[999];
        
        detect(image.vec.data(), image.w, image.h, conf_threshold, objects, object_count, iou_threshold);

        end = clock();
        printf("predict takes %.3f seconds\n", (float)(end-start) / CLOCKS_PER_SEC);

        if (object_count < 1)
        {
            printf("Skip saving, since no object in %s\n", filename.c_str());
            continue;
        }

        ai::save_image(ai::img2mat(image.vec.data(), image.w, image.h), 
            filename, out_dir, model_name);

        save_objects(
            ai::get_out_path(filename, out_dir, model_name).c_str(), 
            objects, 
            object_count);
    }

    de_init(model_path);

    return 0;
}
