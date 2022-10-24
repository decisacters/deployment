
#ifndef NET_API_H
#define NET_API_H

#include "core/_net.h"

#include "daai.h"

int init(const char* model_path, unsigned int num_threads);
int de_init(const char* model_name);
ai::Net* get_net();

#endif // NET_API_H