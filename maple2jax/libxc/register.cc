#include "register.h"

std::map<init_fn, get_param_fn> registry;
std::map<void*, std::string> work_to_maple_name;
