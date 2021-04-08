#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;


int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

Mat src,src_gray;
Mat dst,detected_edges;

static void CannyThreshold(int, void*)
{
//		std::cout << "Callback" << std::endl;
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    imshow( window_name, dst );
}


int main(){

		VideoCapture cap;
		cap.set(CAP_PROP_SETTINGS, 1);
		//		cap.set(CAP_PROP_FRAME_WIDTH, 1020);
		//		cap.set(CAP_PROP_FRAME_HEIGHT, 748);

		if(!cap.open(0)){
				return 0;
		}

		namedWindow(window_name,WINDOW_AUTOSIZE);

		for(;;){
						cap.read(src);
				if( src.empty() ) break; // end of video stream



				// do some processing...
				//

				cvtColor(src,src_gray,COLOR_BGR2GRAY);
				//imshow(window_name,src_gray);


				createTrackbar("Min Threshold", window_name,&lowThreshold,max_lowThreshold,CannyThreshold);

				CannyThreshold(0,0);



				imshow("this is you, smile! :)", src);
				if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
		}


		return 0;
}
