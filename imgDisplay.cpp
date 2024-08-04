/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Image Display with differnet special Effects
*/

#include <cstdio>
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


// Defining the prototypes of methods used in this file
int greyscale( cv::Mat &src, cv::Mat &dst );
int sepiatone( cv::Mat &src, cv::Mat &dst );
int blur5x5_1( Mat &src, Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int sobel3x3SeperableFilter(cv::Mat &src, cv::Mat &dst, int sobelKernel[3], int kernel_multiplier[3]);
int magnitude( Mat &sx, Mat &sy, Mat &dst );
int magnitudeSingleChannel( Mat &sx, Mat &sy, Mat &dst , cv::Mat &angles);
int blurQuantize( Mat &src, Mat &dst, int level );
int stretchImagesH( Mat &frame,Mat &dst );
int cannyEdgeDetection( Mat &src,Mat &dst );
int magnitudeAndAngle( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst, cv::Mat &angles );
int nonMaximaSupression(cv::Mat &src, cv::Mat &dst, cv::Mat &angles);
int hysteresisThresholding( cv::Mat &src, cv::Mat &dst , int upperThreshold, int lowerThreshold);
int brightenContrastImage( cv::Mat &src, cv::Mat &dst, int brightnessValue );
int negativeImage( cv::Mat &src, cv::Mat &dst );

/**
 * @brief Gets the sepia image of the src image.
 * This function takes input a image and finds the 
 * sepia filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getSepiaImage(Mat &src, Mat &dst){
    sepiatone( src, dst );
    return 0;
}

/**
 * @brief Gets the blur image of the src image.
 * This function takes input a image and finds the 
 * blur filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getBlurImage(Mat &src, Mat &dst){
    blur5x5_2( src, dst );
    return 0;
}

/**
 * @brief Gets the xSobel image of the src image.
 * This function takes input a image and finds the 
 * xSobel filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getXSobel(Mat &src, Mat &dst){
    int sobelXKernel[3] = {-1, 0, 1};
    int kernel_multiplier[3] = {1,2,1};
    sobel3x3SeperableFilter(src, dst, sobelXKernel, kernel_multiplier);
    convertScaleAbs(dst, dst);
    return 0;
}

/**
 * @brief Gets the ySobel image of the src image.
 * This function takes input a image and finds the 
 * ySobel filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getYSobel(Mat &src, Mat &dst){
    int sobelYKernel[3] = {1, 2, 1};
    int kernel_multiplier[3] = {1,0,-1};
    sobel3x3SeperableFilter(src, dst, sobelYKernel, kernel_multiplier);
    convertScaleAbs(dst, dst);
    return 0;
}

/**
 * @brief Gets the Sobel magnitude image of the src image.
 * This function takes input a image and finds the 
 * sobel magnitude filtered image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getMSobel(Mat &src, Mat &dst){

    Mat sobelx,sobely;
    int sobelXKernel[3] = {-1, 0, 1};
    int kernelx_multiplier[3] = {1,2,1};
    sobel3x3SeperableFilter(src, sobelx, sobelXKernel, kernelx_multiplier);

    int sobelYKernel[3] = {1, 2, 1};
    int kernely_multiplier[3] = {1,0,-1};
    sobel3x3SeperableFilter(src, sobely, sobelYKernel, kernely_multiplier);

    magnitude( sobelx, sobely, dst );
    convertScaleAbs(dst, dst);
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
int getBlurQuantizedImage(Mat &src, Mat &dst, int level){
    blurQuantize( src, dst, level );
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
int getStretchedImage(Mat &src, Mat &dst){
    stretchImagesH( src,dst);
    return 0;
}

/**
 * @brief Gets the edge detected image of the src image.
 * This function takes input a image and finds the 
 * edges in the image using Canny edge detection.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getCannyEdgeDetetctionImage(Mat &src, Mat &dst){
    Mat gblur,sx,sy, sm,nms, angles;
    cv::GaussianBlur(src, gblur, cv::Size(15, 15), 0);
    getXSobel(gblur, sx);
    cvtColor(sx, sx, COLOR_BGR2GRAY);
    getYSobel(gblur, sy);
    cvtColor(sy, sy, COLOR_BGR2GRAY);
    magnitudeSingleChannel( sx, sy, sm , angles);
    nonMaximaSupression(sm, nms,angles);
    int upperThreshold = 50;
    int lowerThreshold = 20;
    hysteresisThresholding(nms,dst,upperThreshold, lowerThreshold);
    return 0;
}

/**
 * @brief Brightens or contrasts the src image.
 * This function takes input a image and finds the 
 * brightened or contrasted version of the input image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] brightnessLevel the level to which the image should be brightened/contrasted
 * @return 0.
*/
int getBrightContrastImage(Mat &src, Mat &dst, int brightnessLevel){
    brightenContrastImage( src, dst,brightnessLevel );
    return 0;
}

int getNegativeImage(Mat &src, Mat &dst){
    negativeImage( src, dst );
    return 0;
}


int main(int argc, char *argv[]){
    Mat srcImage; //Standard image data type
    char imgFileName[256];

    // command line comments map
    std::unordered_map<string,int> command_map;
    command_map["sepia"] = 1;
    command_map["blur"] = 2;
    command_map["sobelx"] = 3;
    command_map["sobely"] = 4;
    command_map["sobelm"] = 5;
    command_map["blurqz"] = 6;
    command_map["stretch"] = 7;
    command_map["canny"] = 8;
    command_map["brighten"] = 9;
    command_map["negative"] = 10;

    if(argc != 2 && argc != 3){
        printf("Incorrect Command Line input. Usage: ");
        exit(-1);
    }
    strcpy(imgFileName, argv[1]);

    //read the image
    srcImage = imread(imgFileName); // read image fro the given filepath

    //checks if the input image contains data
    if(srcImage.data == NULL) {
        printf("Unable to read image %s\n", imgFileName);
        exit(-1);
    }

    namedWindow(imgFileName, 1);

    Mat dst;
    String imageName, outputImageName;
    int level=0, brightnessLevel=0;
    if(argc == 3){
        outputImageName = argv[2];
        int command = command_map[argv[2]];
        switch(command){
            case 1:
                getSepiaImage(srcImage,dst);
                break;
            case 2:
                getBlurImage(srcImage,dst);
                break;
            case 3: 
                getXSobel(srcImage,dst);
                break;
            case 4:
                getYSobel(srcImage,dst);
                break;
            case 5:
                getMSobel(srcImage,dst);
                break;
            case 6:
                level = 10;
                getBlurQuantizedImage(srcImage,dst,level);
                break;
            case 7:
                getStretchedImage(srcImage,dst);
                break;
            case 8:
                getCannyEdgeDetetctionImage(srcImage,dst);
                break;
            case 9:
                cout<<"Enter the Brightness Intensity Value. Positive value will brighten the image and negative value will contrast the image.\n";
                cin>>brightnessLevel;
                cout<<"Image being processed.";
                getBrightContrastImage(srcImage,dst, brightnessLevel);
                break;
            case 10:
                getNegativeImage(srcImage,dst);
                break;
        }
    }else{
        outputImageName = "Regular";
        srcImage.copyTo(dst);
    }
    string outputImagePath =  "/Users/sachinpc/Documents/cvphotos/saved_photos/" + outputImageName + ".jpg";
    imwrite(outputImagePath, dst); //save the image to a specific location
    imshow(imgFileName, dst); //display the image
    
    
    while(true){
        char ch = waitKey(0);
        if(ch == 'q'){
            break;
        }
    }   
    destroyWindow(imgFileName);

    printf("Terminating\n");

    return(0);
}



