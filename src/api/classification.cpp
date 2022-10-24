
#include "api/classification.h"

int save_class(
    const char* out_path_,
    Object object
)
{
    std::string out_path = std::string(out_path_);
    std::string ext = *(ai::split(out_path, ".").rbegin());
    out_path = ai::replace(out_path, ext, "csv");
    std::ofstream out_file(out_path);
    if (out_file.is_open())
    {
        printf("Save class to %s\n", out_path.c_str());
        out_file << "id,name,score\n";
        out_file << object.id << ","
                << object.name << ","
                << object.score << "\n";
    }
    else
        printf("Could not open %s to save\n", out_path.c_str());
    
    return 0;
}

int classify(
    unsigned char* pixels,
    int image_width,
    int image_height,
    float conf_threshold,
    Object& object
)
{
    std::vector<ai::Mat<unsigned char>> input_mats;
    input_mats.push_back(ai::img2mat(pixels, image_width, image_height));

    ai::Net* net = get_net();

    net->predict(input_mats, conf_threshold, object);

    return 0;
}

int test_classification(
    const char* model_path, 
    const char* in_dir, 
    const char* out_dir, 
    float conf_threshold
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

        clock_t start, end;
        start = clock();
        
        Object object;
        classify(image.vec.data(), image.w, image.h, conf_threshold, object);

        end = clock();
        printf("predict takes %.3f seconds\n", (float)(end-start) / CLOCKS_PER_SEC);

        printf("%s %d %s %f\n", filename.c_str(), object.id, object.name, object.score);

        std::string out_path = ai::get_out_path(filename, out_dir, model_name);
        ai::make_parent_dirs(out_path);

        save_class(out_path.c_str(), object);
    }

    de_init(model_path);

    return 0;
}
