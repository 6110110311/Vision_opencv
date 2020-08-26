#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <opencv2/video/background_segm.hpp>

using namespace std;
using namespace cv;
/// Global Variables
Mat background;
Mat frame;
Mat dst;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
// grad high
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
Mat grad;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
Mat cl;
// morphological
int erosion_elem = 0;
int erosion_size = 22;
int dilation_elem = 0;
int dilation_size = 23;
int const max_elem = 2;
int const max_kernel_size = 40;
Mat elementErosion;
Mat elementDilation;

//object Tracking
vector<vector<Rect>> objs;

void thresh_callback(int, void*);
void Erosion(int, void*);
void Dilation(int, void*);

int main(int argc, char** argv)
{
	VideoCapture cap("C:\\Users\\Ford\\Desktop\\Vision\\Ass_1\\video.avi"); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;

	cap.read(frame);
	Mat acc = Mat::zeros(frame.size(), CV_32FC1);

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		cl = frame;
		//imshow("frame", frame);
		// Get 50% of the new frame and add it to 50% of the accumulator
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		equalizeHist(frame, dst);
		GaussianBlur(dst, frame, Size(21, 21), 0, 0);
		accumulateWeighted(frame, acc, 0.5);
		// Scale it to 8-bit unsigned
		convertScaleAbs(acc, background);
		/*
		imshow("Original", frame);
		imshow("Weighted Average", background);
		*/
		subtract(frame, background, frame);
		threshold(frame, frame, 10, 255, THRESH_BINARY);
		//imshow("Threshold", frame);

		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(frame, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(frame, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		/*
		/// Create Dilation Trackbar
		createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
			&dilation_elem, max_elem,
			Dilation);
		createTrackbar("Kernel size:\n 2n +1", "Dilation Demo",
			&dilation_size, max_kernel_size,
			Dilation);

		// Create Erosion Trackbar
		createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
			&erosion_elem, max_elem,
			Erosion);
		createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
			&erosion_size, max_kernel_size,
			Erosion);
			*/
			/// Default start

		Dilation(0, 0);
		Erosion(0, 0);
		thresh_callback(0, 0);
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in
	return 0;
}

void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	threshold(grad, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours 
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(cl, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);

	}

	/// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", cl);
}

void Erosion(int, void*)
{
	int erosion_type;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	elementErosion = getStructuringElement(erosion_type, Size(45, 45), Point(22, 22));

	/// Apply the erosion operation
	erode(grad, grad, elementErosion);
	dilate(grad, grad, elementDilation);

	//imshow("Erosion Demo", grad);
}

/** @function Dilation */
void Dilation(int, void*)
{
	int dilation_type;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	elementDilation = getStructuringElement(dilation_type, Size(46, 46), Point(23, 23));

	/// Apply the dilation operation
	dilate(grad, grad, elementDilation);
	erode(grad, grad, elementErosion);
	//imshow("Dilation Demo", grad);
}
