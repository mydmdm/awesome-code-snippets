#include "utils.cuh"

int main(int argc, const char *argv[])
{
    int device = 0;
    if (argc == 2)
    {
        device = atoi(argv[1]);
    }
    print_device_properties(device);
}