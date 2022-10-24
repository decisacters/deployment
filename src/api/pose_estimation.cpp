
#include "api/pose_estimation.h"

int max_count_;

std::vector<Object> load_objects(std::string image_path)
{
    std::vector<Object> Objects;

    std::string ext = *(ai::split(image_path, ".").rbegin());
    std::string csv_path = ai::replace(image_path, ext, "csv");

    std::ifstream in_file(csv_path);

    if (in_file.is_open())
    {
        std::string line;
        int line_num = 0;
        std::vector<std::string> headings;
        while (std::getline(in_file, line)) 
        {
            if (line_num != 0)
            {
                std::vector<std::string> items = ai::split(line, ",");
                Object object;
                object.x1 = std::atoi(items[0].c_str());
                object.y1 = std::atoi(items[1].c_str());
                object.x2 = std::atoi(items[2].c_str());
                object.y2 = std::atoi(items[3].c_str());
                object.id = std::atoi(items[4].c_str());
                object.name = (char*)items[5].c_str();
                object.score = 1;
                Objects.push_back(object);
            }
            else
            {
                headings = ai::split(line, ",");
                if (headings[0] != "x1"
                    || headings[1] != "y1"
                    || headings[2] != "x2"
                    || headings[3] != "y2"
                    || headings[4] != "id"
                    || headings[5] != "name"
                )
                {
                    printf("csv heading should be x1,y1,x2,y2,id,name instead of %s\n", line.c_str());
                    return Objects;
                }
            }
            line_num++;
        }
    }
    else
        printf("Could not open %s\n", csv_path.c_str());

    return Objects;
}

int save_points(
    const char* out_path_, 
    Point* points, 
    int* point_count, 
    Object* objects,
    int object_count
)
{
    std::string out_path = std::string(out_path_);
    std::string ext = *(ai::split(out_path, ".").rbegin());
    out_path = ai::replace(out_path, ext, "csv");
    std::ofstream out_file(out_path);
    if (out_file.is_open())
    {
        printf("Save %d points to %s\n", (int)object_count, out_path.c_str());
        out_file << "x1,y1,x2,y2,id,name\n";
        for (size_t i = 0; i < object_count; i++)
        {
            Object object = objects[i];
            out_file << object.x1 << ","
                    << object.y1 << ","
                    << object.x2 << ","
                    << object.y2 << ","
                    << object.id << ","
                    << object.name;

            for (size_t j = 0; j < point_count[i]; j++)
            {
                Point point = points[i*point_count[i]+j];
                out_file << "," << point.x
                        << "," << point.y
                        << "," << point.score;
            }
            out_file << "\n";
        }
    }
    else
        printf("Could not open %s to save\n", out_path.c_str());
    
    return 0;
}

void draw_points(
    unsigned char* pixels, int w, int h, 
    std::map<int, std::vector<short>> color_map,
    Point* points, int point_count, Object object
)
{
    #ifdef OPENCV_FOUND
    cv::Mat mat(h, w, CV_8UC3, pixels);
    cv::Scalar color = cv::Scalar{(double)ai::rand_short(), (double)ai::rand_short(), (double)ai::rand_short()};
    for (size_t i = 0; i < point_count; i++)
    {
                
        float font_scale = (float)((object.x2 - object.x1) * (object.y2 - object.y1)) / (float)w / (float)h;
        font_scale = font_scale < 0.5 ? 0.5f : font_scale;
        int font_pixel_size = (int) (font_scale * 20);
        int thickness = font_pixel_size / 5 + 1;
        int baseline=0;
        int font_face = cv::FONT_HERSHEY_SIMPLEX;

        Point point = points[i];
        int radius = thickness;
        int thickness_ = -1;

        cv::circle(mat, cv::Point(point.x, point.y), radius, color, -1);

        std::string text = std::to_string(i) + " " + std::to_string(point.score).substr(2, 2);

        cv::putText(
            mat, 
            text, 
            cv::Point(point.x, point.y),
            font_face,
            font_scale, 
            color
        );
    }
    #endif
}

int detect(
    unsigned char* pixels,
    int image_width,
    int image_height,
    float conf_threshold,
    Point* points,
    int* point_count,
    Object* objects,
    int object_count
)
{
    std::vector<ai::Mat<unsigned char>> input_mats;
    input_mats.push_back(ai::img2mat(pixels, image_width, image_height));

    std::vector<std::vector<Point>> point_vector;

    std::vector<Object> object_vector;

    // TODO object array to object vector
    for (size_t i = 0; i < object_count; i++)
    {
        Object object;
        object.x1 = objects[i].x1;
        object.y1 = objects[i].y1;
        object.x2 = objects[i].x2;
        object.y2 = objects[i].y2;
        object.id = objects[i].id;
        object.name = objects[i].name;
        object.score = objects[i].score;
        object_vector.push_back(object);
    }

    ai::Net* net = get_net();

    net->predict(input_mats, conf_threshold, point_vector, object_vector);

    for (size_t i = 0; i < point_vector.size(); i++)
    {
        point_count[i] = (int) point_vector[i].size();
        std::copy(point_vector[i].begin(), point_vector[i].end(), &points[i*point_count[i]]);
        draw_points(
            pixels, input_mats[0].w, input_mats[0].h, net->color_map,
            &points[i*point_count[i]], point_count[i], object_vector[i]
        );
    }

    return 0;
}

int test_pose_estimation(
    const char* model_path, 
    const char* in_dir, 
    const char* out_dir, 
    float conf_threshold, 
    int& object_count
)
{
    printf("conf %f\n", conf_threshold);

    if (in_dir == nullptr) return -1;

    std::vector<std::string> filenames = ai::get_images(in_dir);

    if (!filenames.empty() && init(model_path, 1) != 0) return -1;


    for (std::string filename : filenames)
    {

        ai::Mat<unsigned char> image = ai::load_image(filename);

        if (image.vec.empty()) continue;

        std::string model_name = ai::get_model_name(model_path, true);

        int point_count[999];
        Point points[99][999];

        std::vector<Object> objects = load_objects(filename);

        if (objects.empty()) continue;

        clock_t start, end;
        start = clock();
        
        detect(image.vec.data(), image.w, image.h, conf_threshold, 
            &points[0][0], point_count, objects.data(), (int)objects.size());

        end = clock();
        printf("predict takes %.3f seconds\n", (float)(end-start) / CLOCKS_PER_SEC);

        ai::save_image(ai::img2mat(image.vec.data(), image.w, image.h), 
            filename, out_dir, model_name);

        save_points(
            ai::get_out_path(filename, out_dir, model_name).c_str(), 
            &points[0][0], 
            point_count,
            objects.data(), 
            (int)objects.size()
        );
    }

    de_init(model_path);

    return 0;
}
