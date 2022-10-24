
#ifndef DAAI_H
#define DAAI_H

struct Object 
{
    int x1;
    int y1;
    int x2;
    int y2;
    int id;
    char* name;
    float score;
};

struct Point 
{
    int x;
    int y;
    float score;
};

int init(const char* model_path, unsigned int num_threads);
int de_init(const char* model_name);

int test_object_detection(
    const char* model_dir, 
    const char* in_dir, 
    const char* out_dir, 
    float conf_threshold, 
    int& max_count, 
    float iou_threshold
);

int detect(
    unsigned char* pixels, 
    int image_width, 
    int image_height, 
    float conf_threshold, 
    Object* objects, 
    int& object_count, 
    float iou_threshold
);

int test_pose_estimation(
    const char* model_dir, 
    const char* in_dir, 
    const char* out_dir, 
    float conf_threshold, 
    int& max_count
);

int detect(
    unsigned char* pixels, 
    int image_width, 
    int image_height, 
    float conf_threshold, 
    Point* points, 
    int* point_count
);

int test_classification(
    const char* model_dir, 
    const char* in_dir, 
    const char* out_dir, 
    float conf_threshold
);

int classify(
    unsigned char* pixels,
    int image_width,
    int image_height,
    float conf_threshold,
    Object& object
);

#endif // DAAI_H