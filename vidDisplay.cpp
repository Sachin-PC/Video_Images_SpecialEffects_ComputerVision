/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Video Display with differnet special Effects
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "faceDetect.cpp"
// #include "filter.cpp"

using namespace std;
using namespace cv;

// Defining the prototypes of methods used in this file
int greyscale( cv::Mat &src, cv::Mat &dst );
int sepiatone( cv::Mat &src, cv::Mat &dst );
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces );
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale  );
int colourFaces( cv::Mat &frame,cv::Mat &dst, std::vector<cv::Rect> &faces, int minWidth );
int stretchFaces( cv::Mat &frame,cv::Mat &dst, std::vector<cv::Rect> &faces, int minWidth  );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int level );
int sobel3x3SeperableFilter(cv::Mat &src, cv::Mat &dst, int sobelKernel[3], int kernel_multiplier[3]);
int magnitude( Mat &sx, Mat &sy, Mat &dst );


/**
 * @brief Gets the face detected image of the src image.
 * This function takes input a image and detected the 
 * faces in image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getFaceDetectedFrame(cv::Mat &src, cv::Mat &dst){
    cv::Mat grey;
    cvtColor(src, grey, COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);
    detectFaces( grey, faces );
    drawBoxes( src, faces );
    if( faces.size() > 0 ) {
    last.x = (faces[0].x + last.x)/2;
    last.y = (faces[0].y + last.y)/2;
    last.width = (faces[0].width + last.width)/2;
    last.height = (faces[0].height + last.height)/2;
    }
    src.copyTo(dst);
    return 0;
}

/**
 * @brief Gets the face detected  and face coloured image of the src image.
 * This function takes input a image and detects the 
 * faces and only keeps the faces coloured and remaining part greyscale in image.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getColouredFaceFrame(cv::Mat &frame, cv::Mat &dst){
    cv::Mat grey;
    cvtColor(frame, grey, COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);
    detectFaces( grey, faces );
    drawBoxes( frame, faces );
    if( faces.size() > 0 ) {
        last.x = (faces[0].x + last.x)/2;
        last.y = (faces[0].y + last.y)/2;
        last.width = (faces[0].width + last.width)/2;
        last.height = (faces[0].height + last.height)/2;
    }
    colourFaces( frame,dst, faces, 0);
    return 0;
}

/**
 * @brief Gets the face detected  and face zoomed image of the src image.
 * This function takes input a image and detects the 
 * faces and only zoomes the faces and keets the remaining part of image as it is.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int getZoomedFaceFrame(cv::Mat &src, cv::Mat &dst){
    cv::Mat grey;
    cvtColor(src, grey, COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);
    detectFaces( grey, faces );
    // drawBoxes( src, faces );
    if( faces.size() > 0 ) {
    last.x = (faces[0].x + last.x)/2;
    last.y = (faces[0].y + last.y)/2;
    last.width = (faces[0].width + last.width)/2;
    last.height = (faces[0].height + last.height)/2;
    }
    stretchFaces( src,dst, faces, 0 );
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
int getSobelXFrame(cv::Mat &src, cv::Mat &dst){
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
int getSobelYFrame(cv::Mat &src, cv::Mat &dst){
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
int getSobelMagnitudeFrame(cv::Mat &src, cv::Mat &dst){
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

int main(int argc, char** argv) {

    // Load the video file from a specific path
    // VideoCapture cap("/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Project1/Videos/IMG_3601.MOV");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: could not open video file" << endl;
        return -1;
    }

    namedWindow("Video", WINDOW_NORMAL);
    
    bool isGreyscale = false, alternateGreyScale = false, sepiaTone = false, detectFace = false, colourFace = false,warpFace = false;
    int special_effect_id = 0;
    bool quit = false;
    string outputImageName = "Regular";
    string outputImageDir =  "/Users/sachinpc/Documents/cvphotos/saved_video_frames/";
    string outputImagePath;
    Mat frame, dst;

    while (true) {

        cap.read(frame); //same as cap >> frame

        if (frame.empty()) {
            break;
        }

        char ch = waitKey(30); 
        switch(ch){
            case 'q': 
                quit = true;
                break;
            case 's':
                outputImagePath = outputImageDir + outputImageName + ".jpg";
                imwrite(outputImagePath, dst);
                printf("Frame saved succesfully!");
                break;
            case 'n': 
                special_effect_id = 0; 
                outputImageName = "regular";
                break;
            case 'g':
                special_effect_id = 1; 
                outputImageName = "greyscale";
                break;
            case 'h':
                special_effect_id = 2; 
                outputImageName = "alternateGreyscale";
                break;
            case 'i':
                special_effect_id = 3; 
                outputImageName = "blurQuantized";
                break;
            case 'f':
                special_effect_id = 4; 
                outputImageName = "detectFace";
                break;
            case 'c':
                special_effect_id = 5; 
                outputImageName = "colourFace";
                break;
            case 'x':
                special_effect_id = 6; 
                outputImageName = "sobelx";
                break;
            case 'y':
                special_effect_id = 7; 
                outputImageName = "sobely";
                break;
            case 'm':
                special_effect_id = 8; 
                outputImageName = "sobelm";
                break;
            case 'w':
                special_effect_id = 9; 
                outputImageName = "stretchedFace";
                break;

        }        

        if(quit){
            break;
        }
        
        switch(special_effect_id){
            case 0:
                frame.copyTo(dst);
                break;
            case 1:
                cvtColor(frame, dst, COLOR_BGR2GRAY);
                break;
            case 2:
                greyscale(frame, dst);
                break;
            case 3:
                blurQuantize( frame,dst, 10 );
                break;
            case 4:
                getFaceDetectedFrame(frame,dst);
                break;
            case 5:
                getColouredFaceFrame(frame,dst);
                break;
            case 6:
                getSobelXFrame(frame,dst);
                break;
            case 7:
                getSobelYFrame(frame,dst);
                break;
            case 8:
                getSobelMagnitudeFrame(frame,dst);
                break;
            case 9:
                getZoomedFaceFrame(frame,dst);
                break;
        }

        imshow("VideoFrame", dst);
    }

    // Release the video file and destroy the window
    cap.release();
    destroyAllWindows();

    return 0;
}
