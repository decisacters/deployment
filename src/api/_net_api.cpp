#include "api/_net_api.h"

std::string model_path;

ai::Net *net;

int init(const char* model_path, unsigned int num_threads)
{
    int result = 0;

    net = new ai::Net(model_path, num_threads);

    // TODO empty model
    result = std::string(net->model_path).find("empty") == std::string::npos ? 0 : 1;

    return result ? -1 : 0;
}

int de_init(const char* model_path)
{
    if (net != nullptr && std::string(net->model_path).find(model_path) != std::string::npos)
        delete net;
    return 0;
}

ai::Net* get_net()
{
    return net;
}
