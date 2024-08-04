/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Filter logics applied to image frames
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "math.h"

using namespace cv;
using namespace std;

/**
 * @brief Gets the greyscale image of the src image.
 * This function takes input a image and finds the 
 * greyscale filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int greyscale( cv::Mat &src, cv::Mat &dst ){

    //checks if the input image contains data
    if(src.data == NULL) { 
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());

    //updates the pixel values
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            Vec3b pixel = src.at<Vec3b>(i, j);
            int greyValue = (pixel[0] + pixel[1] + pixel[2])/6;
            dst.at<Vec3b>(i, j) = Vec3b(greyValue, greyValue, greyValue);
        }
    }
    return 0;
}

/**
 * @brief Gets the negative image of the src image.
 * This function takes input a image and finds the 
 * negative of the image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int negativeImage( cv::Mat &src, cv::Mat &dst ){

    //checks if the input image contains data
    if(src.data == NULL) { 
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());
    int pixelValue=0;

    //updates the pixel values
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            for(int c=0;c<3;c++){
                pixelValue = src.at<Vec3b>(i, j)[c];
                dst.at<Vec3b>(i, j)[c] = 255 - pixelValue;
            }
        }
    }
    return 0;
}


/**
 * @brief Gets the bright/contrast image of the src image.
 * This function takes input a image and brightens/contrast the 
 * input image by the given brightness value.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] brightnessValue the intensity of brighntenns or contrast(+ve for brigth, -ve for contrast)
 * @return 0.
*/
int brightenContrastImage( cv::Mat &src, cv::Mat &dst, int brightnessValue ){

    //checks if the input image contains data
    if(src.data == NULL) { 
        printf("Unable to read image ");
        exit(-1);
    }
    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());
    int newColourValue =0;

    //updates the pixel values
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            for(int c=0; c< numChannels; c++){
                Vec3b pixel = src.at<Vec3b>(i, j);
                int newColourValue = src.at<Vec3b>(i, j)[c] + brightnessValue;
                if(newColourValue > 255){
                    newColourValue = 255;
                }else if(newColourValue < 0){
                    newColourValue = 0;
                }
                dst.at<Vec3b>(i, j)[c] = newColourValue;
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the sepia image of the src image.
 * This function takes input a image and finds the 
 * sepia filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int sepiatone( cv::Mat &src, cv::Mat &dst ){

    //checks if the input image contains data
    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());
    //checks if the input image size is same as output image
    if (src.size() != dst.size()) {
        return -2;  
    }

    //updates the pixel values
    int r_value =0, g_value=0, b_value=0;
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            Vec3b pixel = src.at<Vec3b>(i, j);
            r_value = 0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0];
            g_value = 0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0];
            b_value = 0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0];
            if(r_value > 255){
                r_value = 255;
            }
            if(g_value > 255){
                g_value = 255;
            }
            if(b_value > 255){
                b_value = 255;
            }
            dst.at<Vec3b>(i, j) = Vec3b(b_value, g_value, r_value);
        }
    }

    return 0;
}

/**
 * @brief Gets the blur image of the src image.
 * This function takes input a image and finds the 
 * blur filtered image. It also considers corned rows and columns
 * and uses a 5*5 kernel matrix.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int blur5x5_0( cv::Mat &src, cv::Mat &dst ){

    //define kernel
    vector<vector<int>> gaussianKernel = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    //checks if the input image contains data
    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());

    //apply kernel to each pixel and update pixel values
    int p=0,q=0,kernel_i=0,kernel_j=0, filter_value=0, filters_sum=0, colour_channel=0;
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
            Vec3b pixel = src.at<Vec3b>(i, j);

            uchar blueValue = pixel[0];
            uchar greenValue = pixel[1];
            uchar redValue = pixel[2];

            p = i-2;
            q = j - 2;
            filter_value = 0;
            filters_sum = 0;
            for(colour_channel=0;colour_channel<3;colour_channel++){
                filters_sum = 0;
                p = i-2;
                filter_value = 0;
                for(kernel_i=0;kernel_i<5;kernel_i++){
                    q = j-2;
                    for(kernel_j=0;kernel_j<5;kernel_j++){
                        if(p <0 || q< 0 || p >= numRows || q >= numCols){
                            // continue;
                        }else{
                            filter_value += gaussianKernel[kernel_i][kernel_j]*src.at<Vec3b>(p, q)[colour_channel];
                            filters_sum +=  gaussianKernel[kernel_i][kernel_j];
                        }
                        q++;
                    }
                    p++;
                }
                filter_value = filter_value/filters_sum;
                if(filter_value > 255 || filter_value < -255){
                    printf("filter value = %d",filter_value);
                }
                dst.at<Vec3b>(i, j)[colour_channel] = filter_value;
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the blur image of the src image.
 * This function takes input a image and finds the 
 * blur filtered image. It uses a 5*5 kernel matrix.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int blur5x5_1( cv::Mat &src, cv::Mat &dst ){

    //define kernel
    vector<vector<int>> gaussianKernel = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    //checks if the input image contains data
    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());
    int p=0,q=0,kernel_i=0,kernel_j=0, filter_value=0, filters_sum=0, colour_channel=0;

    //apply kernel to each pixel and update pixel values
    for(int i=2;i<numRows-2;i++){
        for(int j=2;j<numCols-2;j++){
            for(colour_channel=0;colour_channel<3;colour_channel++){
                filters_sum = 0;
                filter_value = 0;
                p = i-2;
                for(kernel_i=0;kernel_i<5;kernel_i++){
                    q = j-2;
                    for(kernel_j=0;kernel_j<5;kernel_j++){
                        filter_value += gaussianKernel[kernel_i][kernel_j]*src.at<Vec3b>(p, q)[colour_channel];
                        filters_sum +=  gaussianKernel[kernel_i][kernel_j];
                        q++;
                    }
                    p++;
                }
                filter_value = filter_value/filters_sum;
                if(filter_value > 255){
                    filter_value = 255;
                }
                if(filter_value < -255){
                    filter_value = -255;
                }
                dst.at<Vec3b>(i, j)[colour_channel] = filter_value;
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the blur image of the src image.
 * This function takes input a image and finds the 
 * blur filtered image. It uses a 1*5 kernel vector as
 * seperable vectors.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int blur5x5_2( cv::Mat &src, cv::Mat &dst ){

    int gaussianKernel[5] = {1, 2, 4, 2, 1};
    int kernel_multiplier[5] = {1,2,4,2,1};

    //checks if the input image contains data
    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    dst.create(numRows, numCols, src.type());
    int p=0,q=0,kernel_i=0,kernel_j=0, filter_value=0, filters_sum=0, colour_channel=0;
    int pixel_value=0;
    uchar* pixelPointer;

    //apply seperable kernel to each pixel and update pixel values
    for(int i=2;i<numRows-2;i++){
        for(int j=2;j<numCols-2;j++){
            for(colour_channel=0;colour_channel<3;colour_channel++){
                filters_sum = 0;
                filter_value = 0;
                p = i-2;
                for(kernel_i=0;kernel_i<5;kernel_i++){
                    q = j-2;
                    for(kernel_j=0;kernel_j<5;kernel_j++){
                        pixelPointer = src.ptr<uchar>(p, q) + colour_channel;
                        pixel_value = static_cast<int>(*pixelPointer);
                        filter_value += gaussianKernel[kernel_j]*kernel_multiplier[kernel_i]*pixel_value;
                        filters_sum +=  gaussianKernel[kernel_j]*kernel_multiplier[kernel_i];
                        q++;
                    }
                    p++;
                }
                filter_value = filter_value/filters_sum;
                if(filter_value > 255){
                    filter_value = 255;
                }
                if(filter_value < -255){
                    filter_value = -255;
                }
                dst.at<Vec3b>(i, j)[colour_channel] = filter_value;
            }
        }
    }
    return 0;
}


/**
 * @brief Gets the xSobel/ySobel image of the src image.
 * This function takes input a image and finds the 
 * xSobel/ySobel filtered image bsed on kernel passed.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] sobelKernel The filter kernel used as seperable filter.
 * @param[in] kernel_multiplier used with kernel filter to be multiplied with piexl value.
 * @return 0.
*/
int sobel3x3SeperableFilter(cv::Mat &src, cv::Mat &dst, int sobelKernel[3], int kernel_multiplier[3]){
    dst.create(src.size(), CV_16SC3);

    //checks if the input image contains data
    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();

    int p=0,q=0,kernel_i=0,kernel_j=0, filter_value=0, filters_sum=0, colour_channel=0;
    int pixelColourValue=0;

    //apply the kernel given and update the pixel values
    for(int i=1;i<numRows-1;i++){
        for(int j=1;j<numCols-1;j++){
            for(colour_channel=0;colour_channel<3;colour_channel++){
                pixelColourValue = static_cast<int>(src.at<Vec3b>(i, j)[colour_channel]);
                kernel_j = j-1;
                filter_value = 0;
                p = i-1;
                for(kernel_i=0;kernel_i<3;kernel_i++){
                    q = j-1;
                    for(kernel_j=0;kernel_j<3;kernel_j++){
                        filter_value += sobelKernel[kernel_j]*kernel_multiplier[kernel_i]*src.at<Vec3b>(p, q)[colour_channel];
                        q++;
                    }
                    p++;
                }
                dst.at<Vec3s>(i, j)[colour_channel] = filter_value;
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the Sobel magnitude image of the src image.
 * This function takes input a image and finds the 
 * sobel magnitude filtered image.
 * @param[in] sx The SobelX filter image.
 * @param[in] sy The SobelY filter image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){

    dst.create(sx.size(), CV_16SC3);

    //checks if the input image contains data
    if(sx.data == NULL || sy.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = sx.rows;
    int numCols = sx.cols;
    int numChannels = sx.channels();

    int sx_value=0, sy_value = 0, mag_value=0;

    //calculate the eulidian distance and update the pixel values
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
            for(int k =0; k< numChannels; k++){
                sx_value = sx.at<Vec3s>(i, j)[k];
                sy_value = sy.at<Vec3s>(i, j)[k];
                mag_value = sqrt(sx_value*sx_value + sy_value*sy_value);
                dst.at<Vec3s>(i, j)[k] = mag_value;
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the Sobel magnitude image and gradient angle for each pixel of the src image.
 * This function takes input a single channel image and finds the sobel magnitude 
 * and gradient angle for each pixel of the src image.
 * @param[in] sx The SobelX filter image.
 * @param[in] sy The SobelY filter image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] angles gradient angles values for each pixel of the input image.
 * @return 0.
*/
int magnitudeSingleChannel( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst, cv::Mat &angles ){

    //checks if the input images contains data
    if(sx.data == NULL || sy.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = sx.rows;
    int numCols = sx.cols;
    int numChannels = sx.channels();
    dst.create(numRows, numCols, CV_8UC1);
    angles.create(numRows, numCols, CV_64F);
    int redChannelPixelValue = 0;


    int sx_value=0, sy_value = 0, mag_value=0;
    double angle_in_radiance = 0, angle_in_degree=0;

    //calculate the eulidian distance and update the pixel values and also calculate gradient angles
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
                sx_value = static_cast<int>(sx.at<uchar>(i, j));
                sy_value = static_cast<int>(sy.at<uchar>(i, j));
                mag_value = sqrt(sx_value*sx_value + sy_value*sy_value);
                dst.at<uchar>(i, j) = static_cast<uchar>(mag_value);
                angle_in_radiance = atan(sx_value/sy_value);
                angle_in_degree = (angle_in_radiance*180)/M_PI;
                angles.at<double>(i,j) = angle_in_degree;
        }
    }
    return 0;

}

/**
 * @brief Gets the nonMaximaSupression values for the src image.
 * This function takes input a sobel magnitude image,radient angle matrix and performs
 *  nonMaximaSupression
 * @param[in] src The Sobel Magnitude filter image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] angles gradient angles values for each pixel of the input image.
 * @return 0.
*/
int nonMaximaSupressionOld(cv::Mat &src, cv::Mat &dst, cv::Mat &angles){
    int numRows = src.rows;
    int numCols = src.cols;
    double gradientAngle = 0;
    src.copyTo(dst);
    // dst.create(numRows, numCols, CV_8UC1);
    int cur_pixel_value = 0,pixel1_value=0,pixel2_value=0;
    int p1x=0,p1y=0,p2x=0,p2y=0;
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
            gradientAngle = angles.at<double>(i,j);
            if( (gradientAngle >= 0 && gradientAngle <= 22.5) || (gradientAngle > 337.5 && gradientAngle <= 360) || (gradientAngle >= 157.5 && gradientAngle <= 202.5) ){
                // horizontal
                p1x = i;
                p1y = j-1;
                p2x = i;
                p2y = j+1;

            }else if( (gradientAngle > 22.5 && gradientAngle <= 67.5) || (gradientAngle > 202.5 && gradientAngle <= 247.5) ){
                // northeast and south west
                p1x = i-1;
                p1y = j+1;
                p2x = i+1;
                p2y = j-1;
            }else if( (gradientAngle > 67.5 && gradientAngle <= 112.5) || (gradientAngle > 247.5 && gradientAngle <= 292.5) ){
                // vertical
                p1x = i-1;
                p1y = j;
                p2x = i+1;
                p2y = j;
            }else if( (gradientAngle > 112.5 && gradientAngle <= 157.5) || (gradientAngle > 292.5 && gradientAngle <= 237.5) ){
                // northwest and southeast
                p1x = i-1;
                p1y = j-1;
                p2x = i+1;
                p2y = j+1;
            }

            cur_pixel_value = static_cast<int>(src.at<uchar>(i, j));
            if(p1x >= 0 && p1x < (numRows -1) && p1y >= 0 && p1y < numCols){
                pixel1_value = static_cast<int>(src.at<uchar>(p1x, p1y));
            }else{
                pixel1_value = 0;
            }
            if(p2x >= 0 && p2x < (numRows -1) && p2y >= 0 && p2y < numCols){
                pixel2_value = static_cast<int>(src.at<uchar>(p2x, p2y));
            }else{
                pixel2_value = 0;
            }
            if(pixel1_value > cur_pixel_value ||  pixel2_value > cur_pixel_value){
                dst.at<uchar>(i, j) = static_cast<uchar>(0);
            }
        }
    }
    return 0;
}

bool update_pixel_value(int i,int j,int direction, int kernel_size,cv::Mat &src, cv::Mat &dst){

    int start_index = kernel_size/2;
    int p=0,q=0;
    int x=0;
    //compare pixel values with its neighbours based on the gradient angle direction
    //horizontal
    int pixel_value = static_cast<int>(src.at<uchar>(i, j));
    if(direction == 1){
        p = i;
        for(q = j - start_index; q < j + start_index+1; q++){
            if(static_cast<int>(src.at<uchar>(p, q)) > pixel_value){
                x = static_cast<int>(src.at<uchar>(p, q));
                dst.at<uchar>(i, j) = static_cast<uchar>(0);
                return true;
            }
        }
    }
    else if(direction == 2){  //northeast and south west
        q = j + start_index;
        for(p = i - start_index;p<i + start_index + 1; p++){
            if(static_cast<int>(src.at<uchar>(p, q)) > pixel_value){
                x = static_cast<int>(src.at<uchar>(p, q));
                // printf("\n2.static_cast<int>(src.at<uchar>(p, q) = %d and pixel value = %d",x,pixel_value);
                // printf("2222");
                dst.at<uchar>(i, j) = static_cast<uchar>(0);
                return true;
            }
            q--;
        }
    }
    else if(direction == 3){ //vertical
        q = j;
        for(p = i - start_index;p<i + start_index + 1; p++){
            if(static_cast<int>(src.at<uchar>(p, q)) > pixel_value){
                // printf("3333");
                x = static_cast<int>(src.at<uchar>(p, q));
                // printf("\n3.static_cast<int>(src.at<uchar>(p, q) = %d and pixel value = %d",x,pixel_value);
                dst.at<uchar>(i, j) = static_cast<uchar>(0);
                return true;
            }
        }
    }
    else if(direction == 4){  // northwest and southeast
        q = j - start_index;
        for( p = i - start_index;p<i + start_index + 1; p++){
            if(static_cast<int>(src.at<uchar>(p, q)) > pixel_value){
                x = static_cast<int>(src.at<uchar>(p, q));
                // printf("\n4.static_cast<int>(src.at<uchar>(p, q) = %d and pixel value = %d",x,pixel_value);
                dst.at<uchar>(i, j) = static_cast<uchar>(0);
                return true;
            }
            q++;
        }
    }
    return false;
}


/**
 * @brief Gets the nonMaximaSupression values for the src image.
 * This function takes input a sobel magnitude image,radient angle matrix and performs
 *  nonMaximaSupression
 * @param[in] src The Sobel Magnitude filter image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] angles gradient angles values for each pixel of the input image.
 * @return 0.
*/
int nonMaximaSupression(cv::Mat &src, cv::Mat &dst, cv::Mat &angles){
    int numRows = src.rows;
    int numCols = src.cols;
    double gradientAngle = 0;
    src.copyTo(dst);
    // dst.create(numRows, numCols, CV_8UC1);
    int cur_pixel_value = 0,pixel1_value=0,pixel2_value=0;
    int p1x=0,p1y=0,p2x=0,p2y=0;
    int direction=0;
    int kernel_size = 5;
    int start_index = kernel_size/2;

    //determin the direction of angle and update piexl values
    for(int i=start_index;i<numRows-start_index - 1;i++){
        for(int j=start_index;j<numCols - start_index - 1;j++){
            gradientAngle = angles.at<double>(i,j);
            if( (gradientAngle >= 0 && gradientAngle <= 22.5) || (gradientAngle > 337.5 && gradientAngle <= 360) || (gradientAngle >= 157.5 && gradientAngle <= 202.5) ){
                // horizontal
                direction = 1;

            }else if( (gradientAngle > 22.5 && gradientAngle <= 67.5) || (gradientAngle > 202.5 && gradientAngle <= 247.5) ){
                // northeast and south west
                direction = 2;
            }else if( (gradientAngle > 67.5 && gradientAngle <= 112.5) || (gradientAngle > 247.5 && gradientAngle <= 292.5) ){
                // vertical
                direction = 3;

            }else if( (gradientAngle > 112.5 && gradientAngle <= 157.5) || (gradientAngle > 292.5 && gradientAngle <= 237.5) ){
                // northwest and southeast
                direction = 4;
            }

            update_pixel_value(i,j,direction, kernel_size,src,dst);
        }
    }
    return 0;
}


bool setPixelValue(cv::Mat &dst,int i,int j, int numRows, int numCols){
    //CASE 1 - top lEFT
    int px = i-1;
    int py = j-1;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }
    //CASE 2 - top center
    px = i-1;
    py = j;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }
    //CASE 3 - top right
    px = i-1;
    py = j+1;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }
    //CASE 4 - horizontal left
    px = i;
    py = j-1;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }

    //CASE 5 - horizontal center is not required because that is only i,j point

    //CASE 6 - horizontal right
    px = i;
    py = j+1;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }

    //CASE 7 - bottom left
    px = i+1;
    py = j-1;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }

    //CASE 8 - bottom center
    px = i+1;
    py = j;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }

    //CASE 9 - bottom right
    px = i+1;
    py = j+1;
    if(px >=0 && px < numRows-1 && py >=0 && py < numCols){
        if(static_cast<int>(dst.at<uchar>(px, py)) == 255){
            dst.at<uchar>(i, j) = static_cast<uchar>(255);
            return true;
        }
    }
    dst.at<uchar>(i, j) = static_cast<uchar>(0);
    return false;
}

int setPixelValueNew(cv::Mat &dst,int i,int j, int numRows, int numCols, int kernelSize){

    int start_index = kernelSize/2;
    for(int p = i - start_index;p<i + start_index + 1; p++){
        for(int q = j - start_index; q < j + start_index+1; q++){
            if(static_cast<int>(dst.at<uchar>(p, q)) == 255){
                dst.at<uchar>(i, j) = static_cast<uchar>(255);
                return true;
            }
        }
    }
    dst.at<uchar>(i, j) = static_cast<uchar>(0);
    return false;
}

/**
 * @brief Gets the cann edge detected image.
 * This function takes input a nonMaximaSupressed image and then find the canny edge detected image
 * @param[in] src The Sobel Magnitude filter image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] upperThreshold The upper threshold of pixels to be considered.
 * @param[in] lowerThreshold The lower threshold of pixels to be considered.
 * @return 0.
*/
int hysteresisThresholding( cv::Mat &src, cv::Mat &dst , int upperThreshold, int lowerThreshold){

    src.copyTo(dst);
    int numRows = src.rows;
    int numCols = src.cols;
    int cur_pixel_value;

    //update pixel values based on the threshold values
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
            cur_pixel_value = static_cast<int>(src.at<uchar>(i, j));
            if(cur_pixel_value >= upperThreshold){
                dst.at<uchar>(i, j) = static_cast<uchar>(255);
            }else if(cur_pixel_value < lowerThreshold){
                dst.at<uchar>(i, j) = static_cast<uchar>(0);
            }else{
                dst.at<uchar>(i, j) = static_cast<uchar>(127);
            }
        }
    }

    // for(int i=0;i<numRows-1;i++){
    //     for(int j=0;j<numCols;j++){
    //         cur_pixel_value = static_cast<int>(dst.at<uchar>(i, j));
    //         if(cur_pixel_value == 127){
    //             setPixelValue(dst,i,j,  numRows, numCols);
    //         }
    //     }
    // }

    //update the values of the pixels whose value was in between lower and upper threshold
    int kernelSize = 9;
    int start_index = kernelSize/2;
    for(int i=start_index;i<numRows-start_index - 1;i++){
        for(int j=start_index;j<numCols - start_index - 1;j++){
            cur_pixel_value = static_cast<int>(dst.at<uchar>(i, j));
            if(cur_pixel_value == 127){
                setPixelValueNew(dst,i,j,  numRows, numCols, kernelSize);
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the quantized blur image of the src image.
 * This function takes input a image and finds the 
 * quantized blur filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] level the blur intensity level
 * @return 0.
*/
int blurQuantize( cv::Mat &src, cv::Mat &dst, int level ){

    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    dst.create(numRows,numCols, src.type());
    int numChannels = src.channels();

    int pixelColourValue = 0;
    int bucketSize = 255/level;
    int xt=0,xf=0;

    //update the pixel values 
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
            for(int k =0; k< numChannels; k++){
                pixelColourValue = src.at<Vec3b>(i, j)[k];
                xt = pixelColourValue/bucketSize;
                xf = xt*bucketSize;
                dst.at<Vec3b>(i, j)[k] = xf;
            }
        }
    }
    return 0;
}

/**
 * @brief Gets the stretched image of the src image.
 * This function takes input a image and finds the 
 * horizontallly stretched filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int stretchImagesH( cv::Mat &frame,cv::Mat &dst ) {
    cv::Mat frameCopy;
    frame.copyTo(frameCopy);

    int numRows = frame.rows;
    int srcNumCols = frame.cols;
    int numCols = srcNumCols*2;
    int numChannels = frame.channels();

    dst.create(numRows, numCols, frame.type());
    int redChannelPixelValue = 0;
    std::vector<int> columnIndex(numCols, 0);
    for(int j=0;j<numCols;j++){
        columnIndex[j] = j;
    }

    //assign the columns which need to be present at each column of new image
    int center_column = numCols/2;
    int actual_column_center_index = srcNumCols/2;
    int columns_per_block = actual_column_center_index/3;
    int cur_count = 1;
    columnIndex[center_column] = actual_column_center_index;
    int column_index_value = actual_column_center_index -1;
    int i = center_column - 1;
    int per_block_column_count=0;

    // update left column indexes
    while(i>=0){
        for(int j=0;j<cur_count;j++){
            columnIndex[i] = column_index_value;
            if(--i < 0){
                break;
            }
        }
        per_block_column_count++;
        if(per_block_column_count >= columns_per_block){
            per_block_column_count=0;
            cur_count++;
        }
        column_index_value--;
        if(column_index_value < 0){
            column_index_value = 0;
        }
    }

    // update right column indexes
    column_index_value = actual_column_center_index  + 1;
    i = center_column + 1;
    per_block_column_count=0;
    cur_count = 1;
    while(i < numCols){
        for(int j=0;j<cur_count;j++){
            columnIndex[i] = column_index_value;
            if(++i >= numCols){
                break;
            }
        }
        per_block_column_count++;
        if(per_block_column_count >= columns_per_block){
            per_block_column_count=0;
            cur_count++;
        }
        column_index_value++;
        if(column_index_value >= numCols){
            column_index_value = numCols-1;
        }
    }


    //assign pixel values to new image
    for(i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            Vec3b pixel = frameCopy.at<Vec3b>(i, columnIndex[j]);
            dst.at<cv::Vec3b>(i, j) = pixel;
        }
    }
    return 0;
}

/*
int magnitudeAndAngle( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst, cv::Mat &angles ){


    sx.copyTo(dst);
    if(sx.data == NULL || sy.data == NULL) { // No image data read from file
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = sx.rows;
    int numCols = sx.cols;
    int numChannels = sx.channels();
    int redChannelPixelValue = 0;
    if (sx.size() != dst.size()) {
        return -2;  
    }

    double angle_in_radiance = 0, angle_in_degree=0;


    double sx_value=0, sy_value = 0;
    int mag_value=0;
    printf("numRows = %d",numRows);
    cout<<"NUM ROWS = "<<numRows;
    for(int i=0;i<numRows-1;i++){
        printf("i = %d",i);
        for(int j=0;j<numCols;j++){
            for(int k =0; k< numChannels; k++){
                sx_value = sx.at<Vec3s>(i, j)[k];
                sy_value = sy.at<Vec3s>(i, j)[k];
                mag_value = sqrt(sx_value*sx_value + sy_value*sy_value);
                dst.at<Vec3b>(i, j)[k] = mag_value;
            }
        }
    }
    return 0;

}
*/