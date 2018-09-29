#define try_on_depth1
#ifdef try_on_depth

//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>

#include <librealsense2/rs.hpp>



using namespace cv;
using namespace std;
// Global variables
static Mat frame; //current frame
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
int keyboard; //input from keyboard
void help();
void processVideo(char* videoFilename);
void processImages(char* firstFrameFilename);
void help()
{
	cout
		<< "--------------------------------------------------------------------------" << endl
		<< "This program shows how to use background subtraction methods provided by " << endl
		<< " OpenCV. You can process both videos (-vid) and images (-img)." << endl
		<< endl
		<< "Usage:" << endl
		<< "./bs {-vid <video filename>|-img <image filename>}" << endl
		<< "for example: ./bs -vid video.avi" << endl
		<< "or: ./bs -img /data/images/1.png" << endl
		<< "--------------------------------------------------------------------------" << endl
		<< endl;
}
int main(int argc, char* argv[])
{


	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

	//color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2);
	color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0.f);	//fixed scale


	rs2::align align_to(RS2_STREAM_COLOR);


	//Contruct a pipeline which abstracts the device
	rs2::pipeline pipe;

	//Create a configuration for configuring the pipeline with a non default profile
	rs2::config cfg;

	//Add desired streams to configuration
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

	//Instruct pipeline to start streaming with the requested configuration
	pipe.start(cfg);

	// Camera warmup - dropping several first frames to let auto-exposure stabilize
	rs2::frameset frame_set;













	//print help information
	help();
	//check for the input parameter correctness
	if (argc != 3) {
		cerr << "Incorret input list" << endl;
		cerr << "exiting..." << endl;
		//return EXIT_FAILURE;
	}
	//create GUI windows
	namedWindow("Frame");
	namedWindow("FG Mask MOG 2");
	//create Background Subtractor objects
	//pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
	pMOG2 = createBackgroundSubtractorMOG2(500, 16); //MOG2 approach
	//processVideo("0");

	

	//create the capture object
	//VideoCapture capture(0);
	//capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

	//if (!capture.isOpened()) {
	//	//error in opening the video input
	//	cerr << "Unable to open video file: " << "0" << endl;
	//	exit(EXIT_FAILURE);
	//}

	//read input data. ESC or 'q' for quitting
	while (1) {


		frame_set = pipe.wait_for_frames();

		//align depth and color
		//rs2::frameset aligned_frame_set = align_to.process(frame_set);
		//rs2::frameset aligned_frame_set = frame_set;

		//rs2::depth_frame rs_depth_frame = color_map(frame_set.get_depth_frame());
		rs2::depth_frame rs_depth_frame = frame_set.get_depth_frame();
		cv::Mat cv_depth_frame(cv::Size(rs_depth_frame.get_width(), rs_depth_frame.get_height()), CV_16UC1, (void*)rs_depth_frame.get_data(), cv::Mat::AUTO_STEP);
		imshow("depth", cv_depth_frame);

		//normalize(cv_depth_frame, cv_depth_frame, 0, 255, NORM_MINMAX, CV_8UC1);
		//imshow("depth normalize", cv_depth_frame);

		//cv_depth_frame.convertTo(cv_depth_frame, CV_8U);
		//cv_depth_frame.convertTo(cv_depth_frame, CV_8U, 255.0 / 4096.0);
		//imshow("depth convert", cv_depth_frame);

		int nose_distance = 1200;
		int MAX_DETECTION_DISTANCE = 3000; //mm
		int distance_threshold = min(nose_distance, MAX_DETECTION_DISTANCE);

		cv::threshold(cv_depth_frame, cv_depth_frame, distance_threshold, 0, CV_THRESH_TOZERO_INV);
		cv_depth_frame.convertTo(cv_depth_frame, CV_8U, 255.0 / MAX_DETECTION_DISTANCE);
		imshow("depth after threshold", cv_depth_frame);
		//cv_depth_frame.convertTo(cv_depth_frame, CV_8U);
		//imshow("depth after convertto", cv_depth_frame);


		int blurSize = 5;
		int elementSize = 5;
		cv::medianBlur(cv_depth_frame, cv_depth_frame, blurSize);
		imshow("after median blur", cv_depth_frame);




		rs2::video_frame rs_color_frame = frame_set.get_color_frame();
		cv::Mat cv_color_frame(cv::Size(rs_color_frame.get_width(), rs_color_frame.get_height()), CV_8UC3, (void*)rs_color_frame.get_data(), cv::Mat::AUTO_STEP);
		//imshow("input depth", cv_depth_frame);







		//read the current frame


		//update the background model
		pMOG2->apply(cv_depth_frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		stringstream ss;
		rectangle(cv_depth_frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		//ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(cv_depth_frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", cv_depth_frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = waitKey(30);
	}
	//delete capture object
	//capture.release();

	







	/*
	if (strcmp(argv[1], "-vid") == 0) {
	//input data coming from a video
	processVideo(argv[2]);
	}
	else if (strcmp(argv[1], "-img") == 0) {
	//input data coming from a sequence of images
	processImages(argv[2]);
	}
	else {
	//error in reading input parameters
	cerr << "Please, check the input parameters." << endl;
	cerr << "Exiting..." << endl;
	return EXIT_FAILURE;
	}
	*/
	//destroy GUI windows
	destroyAllWindows();
	return EXIT_SUCCESS;
}
void processVideo(char* videoFilename) {
	//create the capture object
	VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27) {
		//read the current frame
		if (!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = waitKey(30);
	}
	//delete capture object
	capture.release();
}
void processImages(char* fistFrameFilename) {
	//read the first file of the sequence
	frame = imread(fistFrameFilename);
	if (frame.empty()) {
		//error in opening the first image
		cerr << "Unable to open first image frame: " << fistFrameFilename << endl;
		exit(EXIT_FAILURE);
	}
	//current image filename
	string fn(fistFrameFilename);
	//read input data. ESC or 'q' for quitting
	while ((char)keyboard != 'q' && (char)keyboard != 27) {
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		//get the frame number and write it on the current frame
		size_t index = fn.find_last_of("/");
		if (index == string::npos) {
			index = fn.find_last_of("\\");
		}
		size_t index2 = fn.find_last_of(".");
		string prefix = fn.substr(0, index + 1);
		string suffix = fn.substr(index2);
		string frameNumberString = fn.substr(index + 1, index2 - index - 1);
		istringstream iss(frameNumberString);
		int frameNumber = 0;
		iss >> frameNumber;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask MOG 2", fgMaskMOG2);
		//get the input from the keyboard
		keyboard = waitKey(30);
		//search for the next image in the sequence
		ostringstream oss;
		oss << (frameNumber + 1);
		string nextFrameNumberString = oss.str();
		string nextFrameFilename = prefix + nextFrameNumberString + suffix;
		//read the next frame
		frame = imread(nextFrameFilename);
		if (frame.empty()) {
			//error in opening the next image in the sequence
			cerr << "Unable to open image frame: " << nextFrameFilename << endl;
			exit(EXIT_FAILURE);
		}
		//update the path of the current frame
		fn.assign(nextFrameFilename);
	}
}

#endif