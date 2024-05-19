#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include<math.h>

__global__ void sobel(unsigned char *inputImage, unsigned char *output, int width, int height,float *gradientMagnitude) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sobel_x[5][5] = { {-1,-2,0,2,1},{-4,-8,0,8,4},{-16,-12,0,12,16},{-4,-8,0,8,4},{-1,-2,0,2,1}};
        int sobel_y[5][5] = { {-1,-4,-6,-4,-1},{-2,-8,-12,-8,-2},{0,0,0,0,0},{2,8,12,8,2},{1,4,6,4,1}};
        int sobel_d[5][5] = { {-6,-4,-1,-2,0},{-4,-12,-8,0,2},{-1,-8,0,8,1},{-2,0,8,12,4},{0,2,1,4,6}};
        int sobel_dt[5][5] = { {0,-2,-1,-4,-6},{2,0,-8,-12,-4},{1,8,0,-8,-1},{4,12,8,0,-2},{6,4,1,2,0}};

        float gradient_x = 0.0;
        float gradient_y = 0.0;
        float gradient_d = 0.0;
        float gradient_dt = 0.0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
              if(x+i>=0 && x+i<width && y+j>=0 && y+j<height){
                gradient_x += sobel_x[i + 2][j + 2] * inputImage[(y + i) * width + (x + j)];
                gradient_y += sobel_y[i + 2][j + 2] * inputImage[(y + i) * width + (x + j)];
                gradient_d += sobel_d[i + 2][j + 2] * inputImage[(y + i) * width + (x + j)];
                gradient_dt += sobel_dt[i + 2][j + 2] * inputImage[(y + i) * width + (x + j)];
              }
            }
        }

        gradientMagnitude[y * width + x] = sqrtf((gradient_x * gradient_x) + (gradient_y * gradient_y) + (gradient_d * gradient_d) + (gradient_dt * gradient_dt));
        if(gradientMagnitude[y * width + x] > 2500){
          output[y*width+x]=255;
        }
        else{
          output[y*width+x]=0;
        }
    }
}

int main() {
    cv::Mat image = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE); //change input file name
    if (image.empty()) {
        printf("Error: Unable to load image.\n");
        return -1;
    }
    int width = image.cols;
    int height = image.rows;

    unsigned char *inputImage, *outputImage;
    inputImage = image.data;
    outputImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    unsigned char *input, *output;
    cudaMalloc((void**)&input, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&output, width*height*sizeof(unsigned char));

    float *magnitude;
    cudaMalloc((void**)&magnitude, width*height*sizeof(float));

    cudaMemcpy(input, inputImage, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    sobel<<<gridDim, blockDim>>>(input, output, width, height, magnitude);

    cudaError_t cudaErrSync = cudaGetLastError();
    if (cudaErrSync != cudaSuccess) {
        printf("CUDA Error (Sync): %s\n", cudaGetErrorString(cudaErrSync));
        return -1;
    }

    cudaMemcpy(outputImage, output, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat processedImage(height, width, CV_8UC1, outputImage);
    cv::imwrite("output5.jpg", processedImage); //change output file name

    cudaFree(input);
    cudaFree(output);
    free(outputImage);

    return 0;
}
