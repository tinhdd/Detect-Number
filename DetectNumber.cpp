#include"stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; 
Mat src_gray;
int thresh = 100;
int max_thresh = 256;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );

int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread("C:\\input.bmp", 1 );

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  imshow("contours", src );

  Mat threshold_output;
  Mat threshold_output1;

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  vector<vector<Point> > contours1;
  vector<Vec4i> hierarchy1;

  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  adaptiveThreshold(src_gray, threshold_output1 ,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY,5,-5);

  int dilation_size = 2;
  Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  dilate( threshold_output1, threshold_output1 , element );

  imshow("threshold",threshold_output);
  imshow("threshold1",threshold_output1);

  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  findContours( threshold_output1, contours1, hierarchy1, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );

  vector<vector<Point> > contours_poly1( contours1.size() );
  vector<Rect> boundRect1( contours1.size() );

  for( int i = 0; i < contours.size(); i++ )
  { 
	  approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
	  boundRect[i] = boundingRect( Mat(contours_poly[i]) );
  }

  for( int i = 0; i < contours1.size(); i++ )
     { 
		 approxPolyDP( Mat(contours1[i]), contours_poly1[i], 3, true );
         boundRect1[i] = boundingRect( Mat(contours_poly1[i]) );
     }

   Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

  int kcx[200];
  int kcy[200];
  for(int i=0 ; i<200 ; i++ )
  { 
	  kcx[i]=0;
	  kcy[i]=0;
  }

  for( int i = 0; i< contours.size(); i++ )
  {
	  kcx[i]=abs( boundRect[i].br().x-boundRect[i].tl().x );
	  kcy[i]=abs( boundRect[i].br().y-boundRect[i].tl().y );
  }

  for( int i=0 ; i< contours.size() ; i++ )
  {
	  if( (kcx[i]/kcy[i])>=3 && (kcx[i]/kcy[i])<=5 && kcx[i] >100 ){
		  kcx[i]=1;
		  kcy[i]=1;
	  }
	  else{
		  kcx[i]=0;
		  kcy[i]=0;
	  }
  }

  Point goc1;
  Point goc2;
  Mat new_img;

   for( int i = 0; i< contours.size(); i++ )
     {
       if( kcx[i]==1 && kcy[i]==1 )
	   {
		    rectangle(src, boundRect[i].tl(), boundRect[i].br(),Scalar(0,0,0), 2, 8, 0 );
			
			goc1.x=boundRect[i].tl().x; 
			goc1.y=boundRect[i].tl().y;

			goc2.x= boundRect[i].br().x;
			goc2.y= boundRect[i].br().y;
	   }
     }

   new_img= src(Rect(goc1.x,goc1.y,(goc2.x-goc1.x),(goc2.y-goc1.y)));
   imwrite("D:\\data\\crop.bmp",new_img);

   int k=1;
   for( int i = 0; i< contours1.size(); i++ )
     {
	   if( boundRect1[i].tl().x > goc1.x && boundRect1[i].tl().y > goc1.y
		   && boundRect1[i].br().x < goc2.x && boundRect1[i].br().y < goc2.y )
	   {
		   if( abs( boundRect1[i].br().y - boundRect1[i].tl().y ) < abs( goc2.y-goc1.y )/2 
			   && abs( boundRect1[i].br().y - boundRect1[i].tl().y ) > abs( goc2.y-goc1.y )/4 )
		   {
				rectangle(src, boundRect1[i].tl(), boundRect1[i].br(), Scalar(255,0,0), 2, 8, 0 );
				new_img= src(Rect(boundRect1[i].tl().x, boundRect1[i].tl().y, boundRect1[i].br().x-boundRect1[i].tl().x, boundRect1[i].br().y-boundRect1[i].tl().y));
				
				char filename[50]="C:\\data\\00.bmp";
				int l=strlen(filename);
				filename[l-5]=(char)(k%10+48);
				filename[l-6]=(char)(k/10+48);
				imwrite(filename,new_img);
				k++;
		   }
	   }
     }

  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", src );

  waitKey(0);
  return(0);
}
