import os
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

import cv2 as cv
import math
from timeit import default_timer as timer
BLOCK_SIZE = 32


# you need to specify the path to cl.exe if it's not in the PATH variable when you installed C++
if (os.system("cl.exe")):
    os.environ['PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"


def grayscale_gpu(img):
    result = numpy.empty_like(img)
    # gets the RGB channels
    R = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    B = img[:, :, 2].copy()

    height = img.shape[0]
    width = img.shape[1]

    # gets the dimensions of the block
    dim_gridx = math.ceil(width / BLOCK_SIZE)
    dim_gridy = math.ceil(height / BLOCK_SIZE)

    # Allocate memory on the gpu
    dev_R = cuda.mem_alloc(R.nbytes)
    dev_G = cuda.mem_alloc(G.nbytes)
    dev_B = cuda.mem_alloc(B.nbytes)

    # copy to gpu
    cuda.memcpy_htod(dev_R, R)
    cuda.memcpy_htod(dev_G, G)
    cuda.memcpy_htod(dev_B, B)

    # Each thread will compute its corresponding cell
    mod = SourceModule("""
            __global__ void ConvertToGray(unsigned char * R, unsigned char * G, unsigned char * B, const unsigned int width, const unsigned int height)
            {
                // Calculate indexes of each thread
                const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
                const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
                
                //if the current thread idx is outside the img boundaries
                if (row >= height && col >= width) {
                    return;
                }
                __shared__ unsigned char R_shared[32][32];
                __shared__ unsigned char G_shared[32][32];
                __shared__ unsigned char B_shared[32][32];
                
                const unsigned int idx = col+row*width; // each thread computes it's working index
                
                // copy data to shared memory
                // each thread copies its corresponding data to the shared memory
                R_shared[threadIdx.y][threadIdx.x] = R[idx];
                G_shared[threadIdx.y][threadIdx.x] = G[idx];
                B_shared[threadIdx.y][threadIdx.x] = B[idx];
                
                const unsigned char intensity = R_shared[threadIdx.y][threadIdx.x]*0.07+G_shared[threadIdx.y][
                threadIdx.x]*0.72+B_shared[threadIdx.y][threadIdx.x]*0.21; 

                R_shared[threadIdx.y][threadIdx.x] = intensity;
                G_shared[threadIdx.y][threadIdx.x] = intensity;
                B_shared[threadIdx.y][threadIdx.x] = intensity;
                
                // copy data back to global memory
                 R[idx] = R_shared[threadIdx.y][threadIdx.x];
                 G[idx] = G_shared[threadIdx.y][threadIdx.x];
                 B[idx] = B_shared[threadIdx.y][threadIdx.x];

                 }
       """)

    grayConv = mod.get_function("ConvertToGray")

    # determine necessary block count
    # 1 block can handle max 1024 threads
    # grid =collection of blocks
    #  cannot communicate to each other
    # block have limited shared memory, which is faster than global memory
    block_count = (height * width - 1) / BLOCK_SIZE * BLOCK_SIZE + 1  # 921600 threads
    grayConv(dev_R,
             dev_G,
             dev_B,
             numpy.uint32(width),
             numpy.uint32(height),
             block=(BLOCK_SIZE, BLOCK_SIZE, 1),  # 1 block has 32*32 = 1024 threads
             grid=(dim_gridx, dim_gridy)
             # Collection of blocks 40*23 -> so it means that we have 40*23*1024 =942080 threads
             )  # 942080-921600 = 20480 thread is unnecessary

    # for example:
    # img shape: 900*1600 so, it has = 1440000 pixels
    # 1440000/1024 = 1407 rounded -> this would be the block count

    # copy result from gpu
    R_new = numpy.empty_like(R)
    cuda.memcpy_dtoh(R_new, dev_R)

    G_new = numpy.empty_like(G)
    cuda.memcpy_dtoh(G_new, dev_G)

    B_new = numpy.empty_like(B)
    cuda.memcpy_dtoh(B_new, dev_B)

    result[:, :, 0] = R_new
    result[:, :, 1] = G_new
    result[:, :, 2] = B_new

    #cannyimg = cv.Canny(result, 100, 200)  # use canny edge detection on the grayscale img
    return result


if __name__ == '__main__':
    image_path = 'landscape.jpg'
    img = cv.imread(image_path)

    timer_start = timer()
    gray_img = grayscale_gpu(img)
    timer_stop = timer()

    print(f'GPU time: {timer_stop - timer_start} seconds')

    timer_start = timer()
    cpu_gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    timer_stop = timer()

    print(f'CPU time: {timer_stop - timer_start} seconds')


    cv.imshow('Image', gray_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
