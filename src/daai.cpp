
#include "daai.h"
#include <cstring>
#include <cstdio>

int main(int argc, char **argv)
{
    if (argc < 4) 
    {
        printf("argc: %d\ntask, model_path, in_dir, out_dir", argc);
        return 1;
    }

    char* task = argv[1];
    char* model_path = argv[2];
    char* in_dir = argv[3];
    char* out_dir = argv[4];

    printf("task: %s\nmodel_path: %s\nin_dir: %s\nout_dir: %s\n", task, model_path, in_dir, out_dir);

    if (!strcmp(task, "object_detection"))
    {
        int max_count = 999;
        float conf_threshold = .25f;
        float iou_threshold = .45f;
        test_object_detection(model_path, in_dir, out_dir, conf_threshold, max_count, iou_threshold);
    }
    else if (!strcmp(task, "pose_estimation"))
    {
        int max_count = 99;
        float conf_threshold = .25f;
        test_pose_estimation(model_path, in_dir, out_dir, conf_threshold, max_count);
    }
    else if (!strcmp(task, "classification"))
    {
        float conf_threshold = .25f;
        test_classification(model_path, in_dir, out_dir, conf_threshold);
    }
    else
        printf("No task match, please enter one of the following:\nobject_detection\npose_estimation\nclassification");

    return 0;
}