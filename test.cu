#include <iostream>
#include <cuda_runtime.h>

template<typename T>
void add(T *a ,T *b, T *c){
    *c=*a+*b;
}

int main() {
    int device = 0;  // 设备编号，可以根据实际情况进行修改

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int maxBlocks, maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxBlocks, cudaDevAttrMaxGridDimX, device);
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);

    std::cout << "Maximum number of blocks: " << maxBlocks << std::endl;
    std::cout << "Maximum threads per block: " << maxThreadsPerBlock << std::endl;


    unsigned short a=2.1;
    float b=1.5;
    int c=a*b;
    printf("%d",c);
    return 0;
}