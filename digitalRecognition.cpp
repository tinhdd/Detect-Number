#include"stdafx.h"
#include"thuvien.h"
#include"window.h"

//// Global variables
int nTrainFaces = 0; // number of training images
int nEigens = 0; // number of eigenvalues
IplImage ** faceImgArr = 0; // array of face images
CvMat * personNumTruthMat = 0; //array of person numbers
IplImage * pAvgTrainImg = 0; // the average image
IplImage ** eigenVectArr = 0; // eigenvectors
CvMat * eigenValMat = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces

Mat src; 
Mat src_gray;
int thresh = 100;
int max_thresh = 255;

int cut_img(void);

//// Function prototypes
void learn();
void recognize();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();

//////////////////////
void SURF_TEST();
void tessractOCR(void);
//////////////////////

void menu(int x,int y);
void menuchon(int x,int y,int select);
int luachon(int x,int y);

int main()
{
Home:
     textbackground(1);
	 system("cls");
	 int cn=luachon(15,5);
	 int cn1,cn2;
	 int mk;
     switch (cn)
     {
		case 1: textbackground(1);system("cls"); SURF_TEST();
				system("pause"); goto Home; break;
		case 2: textbackground(1);system("cls"); recognize();  
				system("pause"); goto Home; break;
		case 3: textbackground(1);system("cls"); tessractOCR(); 
				system("pause"); goto Home; break;
		case 4: textbackground(1);system("cls");  cut_img();
				system("pause"); goto Home; break;
		case 5: system("cls");break;
		}	
}//ket thuc chuong trinh


///////////////////eigenface//////////////

void printUsage()
{
	 printf("Usage: eigenface <command>\n",
	 "  Valid commands are\n"
	 "    train\n"
	 "    test\n");
}
void learn()
{
	 int i;
 
	 // load training data
	 nTrainFaces = loadFaceImgArray("train.txt");
	 if( nTrainFaces < 2 )
	 {
		 fprintf(stderr,
		 "Need 2 or more training faces\n"
		 "Input file contains only %d\n", nTrainFaces);
		 return;
	 }
 
	 // do PCA on the training faces
	 doPCA();
 
	 // project the training images onto the PCA subspace
	 projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
	 for(i=0; i<nTrainFaces; i++)
	 {
		 cvEigenDecomposite(
		 faceImgArr[i],
		 nEigens,
		 eigenVectArr,
		 0, 0,
		 pAvgTrainImg,
		 projectedTrainFaceMat->data.fl + i*nEigens);
	 }
	 // store the recognition data as an xml file
	 storeTrainingData();
}
int loadFaceImgArray(char * filename)
{
	 int sokitu=0;
	 sokitu=cut_img();

	 FILE * imgListFile = 0;
	 char imgFilename[512];
	 int iFace, nFaces=0;
 
	 // open the input file
	 imgListFile = fopen(filename, "r");
 
	 // count the number of faces
	 while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	 rewind(imgListFile);
 
	 // allocate the face-image array and person number matrix
	 faceImgArr = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	 personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );
 
	 // store the face images in an array
	 for(iFace=0; iFace<nFaces; iFace++)
	 {
		 // read person number and name of image file
		 fscanf(imgListFile,
		 "%d %s", personNumTruthMat->data.i+iFace, imgFilename);

		 IplImage* binary=cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
		 cvAdaptiveThreshold(binary,binary,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY_INV,45,-31);

		 // load the face image
		 faceImgArr[iFace] =binary;// cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		 //cvShowImage("binary",faceImgArr[iFace]);
		 //cvWaitKey(0);
	 }
    
	 fclose(imgListFile);
 
	 return nFaces;
}
void doPCA()
{
	 int i;
	 CvTermCriteria calcLimit;
	 CvSize faceImgSize;
 
	 // set the number of eigenvalues to use
	 nEigens = nTrainFaces-1;
 
	 // allocate the eigenvector images
	 faceImgSize.width = faceImgArr[0]->width;
	 faceImgSize.height = faceImgArr[0]->height;
	 eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	 for(i=0; i<nEigens; i++)
	 eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
 
	 // allocate the eigenvalue array
	 eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );
 
	 // allocate the averaged image
	 pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
 
	 // set the PCA termination criterion
	 calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);
 
	 // compute average image, eigenvalues, and eigenvectors
	 cvCalcEigenObjects(nTrainFaces,
	 (void*)faceImgArr,
	 (void*)eigenVectArr,
	 CV_EIGOBJ_NO_CALLBACK,
	 0,
	 0,
	 &calcLimit,
	 pAvgTrainImg,
	 eigenValMat->data.fl);
}
void storeTrainingData()
{
	 CvFileStorage * fileStorage;
	 int i;
 
	 // create a file-storage interface
	 fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );
 
	 // store all the data
	 cvWriteInt( fileStorage, "nEigens", nEigens );
	 cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	 cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	 cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	 cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	 cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	 for(i=0; i<nEigens; i++)
	 {
		 char varname[200];
		 sprintf( varname, "eigenVect_%d", i );
		 cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	 }
 
	 // release the file-storage interface
	 cvReleaseFileStorage( &fileStorage );
}
void recognize()
{
	 int i, nTestFaces = 0; // the number of test images
	 CvMat * trainPersonNumMat = 0; // the person numbers during training
	 float * projectedTestFace = 0;
 
	 // load test images and ground truth for person number
	 nTestFaces = loadFaceImgArray("test.txt");
	 printf("%d test faces loaded\n", nTestFaces);
 
	 // load the saved training data
	 if( !loadTrainingData( &trainPersonNumMat ) ) return;
 
	 // project the test images onto the PCA subspace
	 projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
	 for(i=0; i<nTestFaces; i++)
	 {
		 int iNearest, nearest, truth;
 
		 //project the test image onto the PCA subspace
		 cvEigenDecomposite(
		 faceImgArr[i],
		 nEigens,
		 eigenVectArr,
		 0, 0,
		 pAvgTrainImg,
		 projectedTestFace);
 
		 iNearest = findNearestNeighbor(projectedTestFace);

		 truth = personNumTruthMat->data.i[i];
		 nearest = trainPersonNumMat->data.i[iNearest];
        
		 printf(" nearest = %d, Truth = %d\n", nearest, truth);
	 }
}
 
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	 CvFileStorage * fileStorage;
	 int i;
 
	 // create a file-storage interface
	 fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	 if( !fileStorage )
	 {
		 fprintf(stderr, "Can't open facedata.xml\n");
		 return 0;
	 }
 
	 nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	 nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	 *pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	 eigenValMat = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	 projectedTrainFaceMat =
	 (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	 pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	 eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	 for(i=0; i<nEigens; i++)
	 {
		 char varname[200];
		 sprintf( varname, "eigenVect_%d", i );
		 eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	 }
 
	 // release the file-storage interface
	 cvReleaseFileStorage( &fileStorage );
 
	 return 1;
}
int findNearestNeighbor(float * projectedTestFace)
{
	 double leastDistSq = DBL_MAX;
	 int i, iTrain, iNearest = 0;
 
	 for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	 {
		 double distSq=0;
 
		 for(i=0; i<nEigens; i++)
		 {
			 float d_i =
			 projectedTestFace[i] -
			 projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
			 distSq += d_i*d_i/eigenValMat->data.fl[i];
		 }
 
		 if(distSq < leastDistSq)
		 {
			 leastDistSq = distSq;
			 iNearest = iTrain;
		 }
	 }
 
	 printf("%f", leastDistSq);
	 return iNearest;
}

////////////surf////////////////

void SURF_TEST(void)
{
	ofstream WriteToFile("C:\\surf\\output\\1\\so.txt",ios::trunc);
	char file_name_input[50]="C:\\surf\\input\\1\\1.jpg";
	int l0=strlen(file_name_input);

	for( int i=1 ; i<=8 ; i++)
	{
		file_name_input[l0-5]=(char)(i+48);

	    Mat img_object = imread(file_name_input, CV_LOAD_IMAGE_GRAYSCALE );

		char file_name_data[50]="C:\\surf\\input\\data\\0.jpg";
		int l=strlen(file_name_data);

		int maxp[20];
		for( int i=0 ; i<20 ; i++ ) maxp[i]=0;

		for(int k=0; k<=9 ; k++)
		{
			file_name_data[l-5]=(char)(k+48);

			Mat img_scene = imread(file_name_data, CV_LOAD_IMAGE_GRAYSCALE );

			//-- Step 1: Detect the keypoints using SURF Detector
			int minHessian = 600;

			SurfFeatureDetector detector( minHessian );

			std::vector<KeyPoint> keypoints_object, keypoints_scene;

			detector.detect( img_object, keypoints_object );
			detector.detect( img_scene, keypoints_scene );

			//-- Step 2: Calculate descriptors (feature vectors)
			SurfDescriptorExtractor extractor;

			Mat descriptors_object, descriptors_scene;

			extractor.compute( img_object, keypoints_object, descriptors_object );
			extractor.compute( img_scene, keypoints_scene, descriptors_scene );

			//-- Step 3: Matching descriptor vectors using FLANN matcher
			FlannBasedMatcher matcher;
			std::vector< DMatch > matches;
			matcher.match( descriptors_object, descriptors_scene, matches );

			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for( int i = 0; i < descriptors_object.rows; i++ )
			{ double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
			}

			//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			std::vector< DMatch > good_matches;

			for( int i = 0; i < descriptors_object.rows; i++ )
			{ if( matches[i].distance < 3*min_dist )
				{ good_matches.push_back( matches[i]); }
			}
            
			float orient[730];

			for( int i=0; i< 730 ; i++ )
			{
				orient[i]=0;
			}

			int gt=0;

			for( int i=0 ; i< good_matches.size() ; i++ )
			{
				gt=0;

				gt= keypoints_object[ good_matches[i].queryIdx ].angle -
					keypoints_scene[ good_matches[i].trainIdx ].angle;

				orient[gt+360]++;
			}

			int max=0;
			max = orient[0];

			int xmax=0;

			for( int i=0  ; i<= 730 ; i++ )
			{
				if( max < orient[i] )
				{
					max= orient[i];
					xmax=i;
				}
			}

			for( int i=0 ; i< good_matches.size() ; i++ )
			{
				gt=0;
				gt= keypoints_object[ good_matches[i].queryIdx ].angle -
					keypoints_scene[ good_matches[i].trainIdx ].angle;

				gt=gt+360;

				if( abs( gt- xmax )  > 15 )
				{
					good_matches.erase(good_matches.begin() + i);
					i--;
				}
			}


			if( good_matches.size() <= 1 )
			{
				for( int i=0 ; i < good_matches.size() ; i++ )
				{
					good_matches.erase( good_matches.begin() + i); i--;
				}
			}

			maxp[k]=good_matches.size();
			

			Mat img_matches;
			drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
						good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
			//imshow("son",img_matches);
			//waitKey(0);
		}

        int maxpp=0;
		int so=0;

		for( int k=0 ; k< 20 ; k++){
			if( maxp[k]!=0){
				if( maxpp <= maxp[k])
					{
						maxpp=maxp[k];
						so=k;
				}
			}
		}

		cout<<so;
		
		ofstream WriteToFile("C:\\surf\\output\\1\\so.txt",ios::app);
		WriteToFile <<(char)(so+48);
        WriteToFile <<"\n";
	}

	WriteToFile.close();
}

void tessractOCR(void)
{
	int sokitu=0;

	sokitu=cut_img();

	char file_name[50]="C:\\01.bmp";
	int l=strlen(file_name);

	char kitu[50];
	
	for( int i=1 ; i<sokitu ; i++)
	{
		file_name[l-5]=(char)(i%10+48);
		file_name[l-6]=(char)(i/10+48);

		Mat img;
		img=imread(file_name);
		Mat img_grey;
		cvtColor(img,img_grey,CV_BGR2GRAY);
		Mat binary_img;
		adaptiveThreshold(img_grey, binary_img,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY,45,-15);
	
		Mat element = getStructuringElement( MORPH_RECT,
										   Size( 2*1 + 1, 2*1+1 ),
										   Point( 1, 1 ) );

		dilate( binary_img, binary_img, element );

		Mat element1 = getStructuringElement( MORPH_ELLIPSE,
										   Size( 2*1 + 1, 2*1+1 ),
										   Point( 1, 1 ) );

		erode( binary_img, binary_img, element1 );
		imwrite("1.bmp",binary_img);
		//imshow("seven",binary_img);

		char *outText;

		tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
		api->Init("C:\\Program Files (x86)\\Tesseract-OCR\\tessdata", "seg");
		Pix *image = pixRead("1.bmp");
		api->SetImage(image);
		outText = api->GetUTF8Text();

		//UTF8Text();
		//printf("%s", outText);
		kitu[i]=*outText;
		// Dstroy used object and release memory
		api->End();
		waitKey(0);
	}
	
	for( int i=1 ; i< sokitu ; i++)
	{
		cout<<kitu[i]<<" ";
	}
	cout<<"\n";
}
///////////////////
/////////////////////////////////////////////////////////


/////////////*** cut image***********/

int cut_img(void)
{
	  /// Load source image and convert it to gray
	  src = imread("C:\\input.bmp", 1 );

	  /// Convert image to gray and blur it
	  cvtColor( src, src_gray, CV_BGR2GRAY );
	  blur( src_gray, src_gray, Size(3,3) );

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
	   imwrite("C:\\crop.bmp",new_img);

	   int mang[20];
	   int mang1[20];

	   for( int i=0 ; i< 20 ; i++)
		   {
			   mang[i]=0;
			   mang1[i]=0;
	   }

	   int k=1;
	   for( int i = 0; i< contours1.size(); i++ )
		 {
		   if( boundRect1[i].tl().x > goc1.x && boundRect1[i].tl().y > goc1.y
			   && boundRect1[i].br().x < goc2.x && boundRect1[i].br().y < goc2.y )
		   {
			   if( abs( boundRect1[i].br().y - boundRect1[i].tl().y ) < abs( goc2.y-goc1.y )/2 
				   && abs( boundRect1[i].br().y - boundRect1[i].tl().y ) > abs( goc2.y-goc1.y )/4 )
			   {
					Mat drawing = Mat::zeros( Size(25,35), CV_8UC3 );
					new_img= src(Rect(boundRect1[i].tl().x, boundRect1[i].tl().y, boundRect1[i].br().x-boundRect1[i].tl().x, boundRect1[i].br().y-boundRect1[i].tl().y));
	
					mang[k]=boundRect1[i].tl().x;
					mang1[k]=boundRect1[i].tl().x;

					k++;
			   }
		   }
		 }

	  int temp=0;

	  for( int i=1 ; i< k ; i++)
	  {
		  for( int j=1 ; j< k ; j++)
		  {
			 if( mang1[i] < mang1[j] )
			 {
				 temp=mang1[i];
				 mang1[i]=mang1[j];
				 mang1[j]=temp;
			 }
		  }
	  }

	  for( int i=1 ; i< k ; i++)
	  {
		  for( int j=1 ; j< k ; j++)
		  {
			  if( mang[i]==mang1[j] )
			  {
				  mang[i]=j;
			  }
		  }
	  }

	   k=1;
	   for( int i = 0; i< contours1.size(); i++ )
		 {
		   if( boundRect1[i].tl().x > goc1.x && boundRect1[i].tl().y > goc1.y
			   && boundRect1[i].br().x < goc2.x && boundRect1[i].br().y < goc2.y )
		   {
			   if( abs( boundRect1[i].br().y - boundRect1[i].tl().y ) < abs( goc2.y-goc1.y )/2 
				   && abs( boundRect1[i].br().y - boundRect1[i].tl().y ) > abs( goc2.y-goc1.y )/4 )
			   {
					Mat drawing = Mat::zeros( Size(25,35), CV_8UC3 );
					new_img= src(Rect(boundRect1[i].tl().x, boundRect1[i].tl().y, boundRect1[i].br().x-boundRect1[i].tl().x, boundRect1[i].br().y-boundRect1[i].tl().y));
				
					for( int i=0 ; i< new_img.rows; i++)
					{
						for( int j=0 ; j< new_img.cols; j++)
						{
							drawing.at<Vec3b>(i,j)[0]=new_img.at<Vec3b>(i,j)[0];
							drawing.at<Vec3b>(i,j)[1]=new_img.at<Vec3b>(i,j)[1];
							drawing.at<Vec3b>(i,j)[2]=new_img.at<Vec3b>(i,j)[2];
						}
					}

					char filename[50]="C:\\00.bmp";
					int l=strlen(filename);

					filename[l-5]=(char)( mang[k]%10+48);
					filename[l-6]=(char)( mang[k]/10+48);
					imwrite(filename,drawing);

					k++;
			   }
		   }
		 }
	   return k;
}

////*****************///

void menu(int x,int y)
{
	textbackground(1);
	textcolor(10);
	gotoxy(x,y-1);  printf("SEVEN SEC RECOGNITION ");
	gotoxy(x,y+1);  printf("HAY CHON MOT CHUC NANG:");
	textcolor(12);
	gotoxy(x,y+5);  printf("1. SURF Algorithm         ");
	gotoxy(x,y+7);  printf("2. PCA Algorithm          ");
	gotoxy(x,y+9);  printf("3. ORC Algorithm          ");
	gotoxy(x,y+11); printf("4. Help                   ");
	gotoxy(x,y+13); printf("8. EXIT                   ");
	textcolor(10);
	gotoxy(x,y+17); printf("                            CONG TY V.N.E.X.T.");
	 
}
void menuchon(int x,int y,int select)
{
	switch(select)
	{
	
	case 1:
		{
			menu(x,y);
			textbackground(6);
			textcolor(14);
			gotoxy(x,y+5); printf("1. SURF Algorithm         ");
		}break;
	case 2:
		{
			menu(x,y);
			textbackground(6);
			textcolor(14);
			gotoxy(x,y+7); printf("2. PCA Algorithm          ");
		}break;
	case 3:
		{
			menu(x,y);
			textbackground(6);
			textcolor(14);
			gotoxy(x,y+9); printf("3. ORC Algorithm          ");
		}break;
	case 4:
		{
			menu(x,y);
			textbackground(6);
			textcolor(14);
			gotoxy(x,y+11); printf("4. Help                   ");
		}break;
	case 5:
		{
			menu(x,y);
			textbackground(6);
			textcolor(14);
			gotoxy(x,y+13); printf("8. EXIT                   ");
		}break;
	}
}
int luachon(int x,int y)
{
	int thoat=0;
	int i=1;
	while(!thoat)
	{
		menuchon(x,y,i);
		char c=getch();
		if(c==1) c=getch();
		switch(c)
		{
		case 13:
			{
				thoat=1;
				return i;
			}
		case 80:
			{
				if(i==5) i=1;
				else i++;
				menuchon(x,y,i);
			}break;
		case 72:
			{
				if(i==1) i=5;
				else i--;
				menuchon(x,y,i);
			}break;		
		}
	}
}

