/*
  Sachin Palahalli Chandrakumar
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them

  The path to the Haar cascade file is define in faceDetect.h
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"

using namespace cv;
using namespace std;

bool facePixel(int i,int j,std::vector<cv::Rect> &faces, int minWidth);


/*
  Arguments:
  cv::Mat grey  - a greyscale source image in which to detect faces
  std::vector<cv::Rect> &faces - a standard vector of cv::Rect rectangles indicating where faces were found
     if the length of the vector is zero, no faces were found
 */
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces ) {
  // a static variable to hold a half-size image
  static cv::Mat half;
  
  // a static variable to hold the classifier
  static cv::CascadeClassifier face_cascade;

  // the path to the haar cascade file
  static cv::String face_cascade_file(FACE_CASCADE_FILE);

  if( face_cascade.empty() ) {
    if( !face_cascade.load( face_cascade_file ) ) {
      printf("Unable to load face cascade file\n");
      printf("Terminating\n");
      exit(-1);
    }
  }

  // clear the vector of faces
  faces.clear();
  
  // cut the image size in half to reduce processing time
  cv::resize( grey, half, cv::Size(grey.cols/2, grey.rows/2) );

  // equalize the image
  cv::equalizeHist( half, half );

  // apply the Haar cascade detector
  face_cascade.detectMultiScale( half, faces );

  // adjust the rectangle sizes back to the full size image
  for(int i=0;i<faces.size();i++) {
    faces[i].x *= 2;
    faces[i].y *= 2;
    faces[i].width *= 2;
    faces[i].height *= 2;
    // printf("\n1.x = %d, y = %d, width = %d, height = %d",faces[i].x,faces[i].y,faces[i].width,faces[i].height);
  }

  return(0);
}

/* Draws rectangles into frame given a vector of rectangles
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minSize - ignore rectangles with a width small than this argument
   float scale - scale the rectangle values by this factor (in case frame is different than the source image)
 */
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale  ) {
  // The color to draw, you can change it here (B, G, R)
  cv::Scalar wcolor(170, 120, 110);

  for(int i=0;i<faces.size();i++) {
    if( faces[i].width > minWidth ) {
      cv::Rect face( faces[i] );
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      cv::rectangle( frame, face, wcolor, 3 );
      // printf("\n2.x = %d, y = %d, width = %d, height = %d",faces[i].x,faces[i].y,faces[i].width,faces[i].height);
    }
  }

  return(0);
}

/* color faces and keep the remaing part greyscale
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   cv::Mat &dst - iout image with the result
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minWidth - ignore rectangles with a width small than this argument
 */
int colourFaces( cv::Mat &frame,cv::Mat &dst, std::vector<cv::Rect> &faces, int minWidth  ) {


  if(frame.data == NULL) { 
        printf("Unable to read image ");
        exit(-1);
  }

  int numRows = frame.rows;
  int numCols = frame.cols;
  int numChannels = frame.channels();
  dst.create(numRows,numCols,frame.type());
  int redChannelPixelValue = 0;

  //check the size of src and dst. Both should be equal
  if (frame.size() != dst.size()) {
      return -2;  
  }

  //assigne values to pixel based on if its part of face or not
  for(int i=0;i<numRows;i++){
      for(int j=0;j<numCols;j++){
        Vec3b pixel = frame.at<Vec3b>(i, j);
        if(facePixel(i,j,faces, minWidth)){
          dst.at<cv::Vec3b>(i, j) = pixel;
        }else{
          uchar greyValue = 255 - pixel[1];
          dst.at<cv::Vec3b>(i, j) = Vec3b(greyValue, greyValue, greyValue);
        }
      }
  }

  return 0;
}

/* determins if the pixel i,j is within a face detected
   
   Arguments:
   int i - row index of the pixel
   int j - col index of the pixel
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minWidth - ignore rectangles with a width small than this argument
 */
bool facePixel(int i,int j,std::vector<cv::Rect> &faces, int minWidth){

  int x1=0,x2=0,y1=0,y2=0;
  for(int p=0;p<faces.size();p++) {
    if( faces[p].width > minWidth ) {
      x1 = faces[p].x;
      x2 = faces[p].x + faces[p].height;
      y1 = faces[p].y;
      y2 = faces[p].y + faces[p].width;
      if(i >= y1 && i <= y2 && j >= x1 && j <= x2){
        return true;
      }
    }
  }

  // cv::Point pixelpoint(j, i);
  // for(int p=0;p<faces.size();p++) {
  //   if(faces[p].contains(pixelpoint)){
  //     return true;
  //   }
  // }
  return false;
}


/* zooms the faces and keep the remaing part as it is
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   cv::Mat &dst - out image with the result
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minWidth - ignore rectangles with a width small than this argument
 */
int stretchFaces( cv::Mat &frame,cv::Mat &dst, std::vector<cv::Rect> &faces, int minWidth ) {

  cv::Mat frameCopy;
  frame.copyTo(frameCopy);
  frame.copyTo(dst);

  int numRows = frame.rows;
  int numCols = frame.cols;
  int numChannels = frame.channels();
  int redChannelPixelValue = 0;

  //check the size of src and dst. Both should be equal
  if (frameCopy.size() != dst.size()) {
      return -2;  
  }

  //assign the columns which need to be present at each column of new image
  std::vector<int> columnIndex(numCols, 0);
  for(int j=0;j<numCols;j++){
    columnIndex[j] = j;
  }

  int face_center_column = faces[0].x + faces[0].width/2;
  columnIndex[face_center_column] = face_center_column;

  int column_index_value = face_center_column -1;
  int i = face_center_column - 1;
  // update left column indexes
  while(i>=0){
    columnIndex[i] = column_index_value;
    if(--i < 0){
      break;
    }
    columnIndex[i] = column_index_value;
    column_index_value--;
    i--;
    if(i < faces[0].x){
      break;
    }
  }

  // update right column indexes
  column_index_value = face_center_column  + 1;
  i = face_center_column + 1;
  while(i < numCols){
    columnIndex[i] = column_index_value;
    if(++i > numCols){
      break;
    }
    columnIndex[i] = column_index_value;
    column_index_value++;
    i++;
    if(i > (faces[0].x + faces[0].width)){
      break;
    }
  }

  //assign pixel values to new image
  for(i=0;i<numRows;i++){
      for(int j=0;j<numCols;j++){
        if( i >= faces[0].y && i <= (faces[0].y + faces[0].height)){
          Vec3b pixel = frameCopy.at<Vec3b>(i, columnIndex[j]);
          dst.at<cv::Vec3b>(i, j) = pixel;
        }
      }
  }

  return 0;
}


