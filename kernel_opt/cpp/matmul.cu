#include "utils.h"

int main(int argc, char *argv[])
{
    assert_eq(argc, 4, "InputArgs");
    auto M = atoi(argv[1]);
    auto N = atoi(argv[2]);
    auto K = atoi(argv[3]);
    fprintf(stdout, "matmul test with shape (M,N,K)=(%d,%d,%d)\n", M, N, K);
}