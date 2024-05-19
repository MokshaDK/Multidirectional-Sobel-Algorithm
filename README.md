# Multidirectional-Sobel-Algorithm

An improvement on the traditional Sobel Algorithm, designed to enhance detection of diagonal edges. Implemented in CUDA C to parallelize the process, reducing computation time.

## Files

- `two_directional_sobel.cu`: Implements the traditional Sobel Algorithm.
- `multidirectional_sobel.cu`: Implements the enhanced Multidirectional Sobel Algorithm.

## Usage

To run the code, follow these steps:
1. Modify the name of the image in the source code to detect edges in that image.
2. Compile using the following commands:
nvcc two_directional_sobel.cu -o two_directional_sobel -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs
./two_directional_sobel
nvcc multidirectional_sobel.cu -o multidirectional_sobel -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs
./multidirectional_sobel

## Report

A comprehensive report is provided, containing experimental results, comparisons, and analysis.
