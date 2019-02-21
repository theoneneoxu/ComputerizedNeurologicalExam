#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include <fstream>
#include <chrono>

#include <librealsense2/rs.hpp>

using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;





int calibration_time = 8000; //ms


int frame_rate = 60;
int resolution_width = 848;
int resolution_height = 480;
/*
int frame_rate = 30;
int resolution_width = 1280;
int resolution_height = 720;
*/

int face_detection_offset_width = resolution_width / 4;
int face_detection_offset_height = resolution_height / 6;
int face_detaction_width = resolution_width / 2;
int face_detaction_height = resolution_height * 2 / 3;








int minH = 0, maxH = 17, minS = 255, maxS = 255, minV = 255, maxV = 255;


float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{

	float dist1 = std::sqrt((px1 - cx1)*(px1 - cx1) + (py1 - cy1)*(py1 - cy1));
	float dist2 = std::sqrt((px2 - cx1)*(px2 - cx1) + (py2 - cy1)*(py2 - cy1));

	float Ax, Ay;
	float Bx, By;
	float Cx, Cy;

	//find closest point to C  
	//printf("dist = %lf %lf\n", dist1, dist2);  

	Cx = cx1;
	Cy = cy1;
	if (dist1 < dist2)
	{
		Bx = px1;
		By = py1;
		Ax = px2;
		Ay = py2;


	}
	else {
		Bx = px2;
		By = py2;
		Ax = px1;
		Ay = py1;
	}


	float Q1 = Cx - Ax;
	float Q2 = Cy - Ay;
	float P1 = Bx - Ax;
	float P2 = By - Ay;


	float A = std::acos((P1*Q1 + P2*Q2) / (std::sqrt(P1*P1 + P2*P2) * std::sqrt(Q1*Q1 + Q2*Q2)));

	A = A * 180 / CV_PI;

	return A;
}



int main(int argc, const char** argv)
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
	cfg.enable_stream(RS2_STREAM_COLOR, resolution_width, resolution_height, RS2_FORMAT_BGR8, frame_rate);
	cfg.enable_stream(RS2_STREAM_DEPTH, resolution_width, resolution_height, RS2_FORMAT_Z16, frame_rate);

	//Instruct pipeline to start streaming with the requested configuration
	pipe.start(cfg);

	// Camera warmup - dropping several first frames to let auto-exposure stabilize
	rs2::frameset frame_set;


	

	std::ofstream outputFile;
	outputFile.open("Finger Tracking.txt");

	using namespace std::chrono;
	milliseconds start_time = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);





	//OpenCV Cascade
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|haarcascade_frontalface_alt.xml|Path to face cascade.}"
		//"{face_cascade|lbpcascade_frontalface.xml|Path to face cascade.}"
		//"{face_cascade|Hand.Cascade.1.xml|Path to face cascade.}"
		"{eyes_cascade|haarcascade_mcs_nose.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();
	String face_cascade_name = parser.get<String>("face_cascade");
	String eyes_cascade_name = parser.get<String>("eyes_cascade");
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return -1;
	};


	int face_distance = 1;
	int th = 1;

	//body part set to block in depth
	cv::Rect torso;

	//fingertip point
	cv::Point fingerPoint;

	bool calibration_flag = false;



	while (true)
	{
		//first few seconds are used to calibrate.
		milliseconds ms = duration_cast< milliseconds >(
			system_clock::now().time_since_epoch()
			);
		calibration_flag = (ms.count() - start_time.count()) < calibration_time;
		


		frame_set = pipe.wait_for_frames();

		//align depth and color
		rs2::frameset aligned_frame_set = align_to.process(frame_set);

		rs2::depth_frame rs_depth_frame = aligned_frame_set.get_depth_frame();
		cv::Mat cv_depth_frame(cv::Size(rs_depth_frame.get_width(), rs_depth_frame.get_height()), CV_16UC1, (void*)rs_depth_frame.get_data(), cv::Mat::AUTO_STEP);

		rs2::video_frame rs_color_frame = aligned_frame_set.get_color_frame();
		cv::Mat cv_color_frame(cv::Size(rs_color_frame.get_width(), rs_color_frame.get_height()), CV_8UC3, (void*)rs_color_frame.get_data(), cv::Mat::AUTO_STEP);









		//Apply the classifier to the frame
		//detectAndDisplay(cv_color_frame);
		Mat frame_gray;
		cvtColor(cv_color_frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		std::vector<Rect> faces;
		Mat small_frame;
		frame_gray;
		Rect region_of_interest = Rect(face_detection_offset_width, face_detection_offset_height, face_detaction_width, face_detaction_height);
		cv::rectangle(cv_color_frame, region_of_interest, cv::Scalar(0, 255, 0));
		Mat roi = frame_gray(region_of_interest);

		//face_cascade.detectMultiScale(frame_gray, faces);
		//face_cascade.detectMultiScale(frame_gray, faces, 1.2, 3, 0, cv::Size(20, 20), cv::Size(200, 200));

		//test for nose
		//too slow to directly detect nose
		//face_cascade.detectMultiScale(roi, faces, 1.1, 3);

		//for face detection
		face_cascade.detectMultiScale(roi, faces, 1.1, 3, 0, cv::Size(resolution_width / 10, resolution_width / 10), cv::Size(resolution_width / 4, resolution_width / 4));

		if (faces.size() > 1) {
			cout << "face size more than 1, it is " << faces.size() << endl;
		}

		for (size_t i = 0; i < faces.size(); i++)
		{
			//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			Point center(faces[i].x + faces[i].width / 2 + face_detection_offset_width, faces[i].y + faces[i].height / 2 + face_detection_offset_height);

			
			//if (fingerPoint.x < center.x - faces[i].width / 2 || fingerPoint.x > center.x + faces[i].width / 2 || fingerPoint.y < center.y - faces[i].height / 2 || fingerPoint.y > center.y + faces[i].height / 2) 
			{


				int face_center_x = center.x;
				int face_center_y = center.y;
				int face_x_d = faces[i].width / 4;
				int face_y_d = faces[i].height / 4;



				int face_distance_sample_count = 0;
				float face_distance_sample_sum = 0;
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						float distance = rs_depth_frame.get_distance(face_center_x + face_x_d * i, face_center_y + face_y_d * j) * 1000;
						if (distance > 0) {
							face_distance_sample_sum += distance;
							face_distance_sample_count++;
						}
					}
				}
				//used average distance of multiple points to be face distance
				int new_face_distance = face_distance_sample_count > 0 ? face_distance_sample_sum / face_distance_sample_count : face_distance;
				double alpha = 0.5;
				face_distance = (1 - alpha) * face_distance + alpha * new_face_distance;
				th = face_distance - 30;

			}

			
			





			//nose point
			circle(cv_color_frame, center, 2, Scalar(255, 0, 255), 4);
			//circle(cv_depth_frame, center, 2, Scalar(255, 0, 255), 4);
			//face
			ellipse(cv_color_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, resolution_height, Scalar(255, 0, 255), 4);
			//ellipse(cv_depth_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);

			//Vec3b& nose_color_hsv = cv_depth_frame.at<Vec3b>(center);
			//maxH = nose_color_hsv[0] > 0 ? nose_color_hsv[0] - 1 : maxH;

			//cout << "maxH = " << maxH << ", nose distance = " << rs_depth_frame.get_distance(center.x, center.y) * 1000 << endl;


			int x = faces[i].x + face_detection_offset_width;
			int y = faces[i].y + face_detection_offset_height;
			int width = faces[i].width;
			int height = faces[i].height;

			Rect noseRect = Rect(x, y, width, height);
			cv::Point pt1(x, y);
			cv::Point pt2(x + width, y + height);

			cv::Point pt3(faces[i].x, faces[i].y);
			cv::Point pt4(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

			Mat faceROI = frame_gray(noseRect);

			//-- In each face, detect eyes
			std::vector<Rect> eyes;

			//eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3);

			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2 + face_detection_offset_width, faces[i].y + eyes[j].y + eyes[j].height / 2 + face_detection_offset_height);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.2);
				circle(cv_color_frame, eye_center, radius, Scalar(255, 0, 0), 4);
			}


		}




		



		//threshold(cv_depth_frame, cv_depth_frame, std::min(th, 2000), 0, CV_THRESH_TOZERO_INV);
		threshold(cv_depth_frame, cv_depth_frame, th, 0, CV_THRESH_TOZERO_INV);

		cv_depth_frame.convertTo(cv_depth_frame, CV_8U);

		cv::rectangle(cv_color_frame, torso, cv::Scalar(255, 255, 255));

		//after calibration, always block body part in depth input
		if (!calibration_flag) {
			cv_depth_frame(torso).setTo(0);
		}





		//cv::inRange(hsv_frame, cv::Scalar(minH, minS, minV), cv::Scalar(maxH, maxS, maxV), hsv_frame);
		//imshow("after inrange", hsv_frame);
		// Pre processing
		int blurSize = 5;
		int elementSize = 5;
		cv::medianBlur(cv_depth_frame, cv_depth_frame, blurSize);
		cv::imshow("after median blur", cv_depth_frame);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * elementSize + 1, 2 * elementSize + 1), cv::Point(elementSize, elementSize));
		cv::dilate(cv_depth_frame, cv_depth_frame, element);
		// Contour detection
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(cv_depth_frame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		
		//remove small contours
		cout << "contour size before remove " << contours.size() << endl;
		//cout << contours.size() << endl;
		auto iterator = contours.begin();
		while (iterator != contours.end()) {
			int size_th = cv_depth_frame.size().width * cv_depth_frame.size().height * 3 / (face_distance + 1);
			cout << size_th << "  area is " << cv::contourArea(*iterator) << endl;
			if (cv::contourArea(*iterator) < size_th) {
				iterator = contours.erase(iterator);
			}
			else {
				iterator++;
			}
		}
		cout << "contour size after remove " << contours.size() << endl;


		size_t largestContour = 0;
		for (size_t i = 1; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > cv::contourArea(contours[largestContour]))
				largestContour = i;
		}
		cv::drawContours(cv_depth_frame, contours, largestContour, cv::Scalar(0, 0, 255), 1);
		
		// Convex hull
		if (!contours.empty())
		{
			std::vector<std::vector<cv::Point>> hull(1);
			cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
			cv::drawContours(cv_depth_frame, hull, 0, cv::Scalar(0, 255, 0), 3);
			if (hull[0].size() > 2)
			{


				std::vector<int> hullIndexes;
				cv::convexHull(cv::Mat(contours[largestContour]), hullIndexes, true);
				std::vector<cv::Vec4i> convexityDefects;
				cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndexes, convexityDefects);
				cv::Rect boundingBox = cv::boundingRect(hull[0]);



				//for now, first few seconds is used to detect torso
				if (calibration_flag) {
					boundingBox.x = 0;
					boundingBox.width = resolution_width;
					int extra = boundingBox.y * 0.03;
					boundingBox.y -= extra;
					boundingBox.height += extra;
					torso = boundingBox;
				}



				cv::rectangle(cv_depth_frame, boundingBox, cv::Scalar(255, 0, 0));
				cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
				std::vector<cv::Point> validPoints;
				for (size_t i = 0; i < convexityDefects.size(); i++)
				{
					cv::Point p1 = contours[largestContour][convexityDefects[i][0]];
					cv::Point p2 = contours[largestContour][convexityDefects[i][1]];
					cv::Point p3 = contours[largestContour][convexityDefects[i][2]];
					double angle = std::atan2(center.y - p1.y, center.x - p1.x) * 180 / CV_PI;
					double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
					double length = std::sqrt(std::pow(p1.x - p3.x, 2) + std::pow(p1.y - p3.y, 2));
					//if (angle > angleMin - 180 && angle < angleMax - 180 && inAngle > inAngleMin - 180 && inAngle < inAngleMax - 180 && length > lengthMin / 100.0 * boundingBox.height && length < lengthMax / 100.0 * boundingBox.height)
					{
						validPoints.push_back(p1);
					}
				}

				fingerPoint.x = rs_depth_frame.get_width();
				fingerPoint.y = rs_depth_frame.get_height();
				for (auto& p : validPoints) {
					if (p.y < fingerPoint.y)
					{
						fingerPoint = p;
					}
				}

				if (!calibration_flag) {
					//draw finger point on depth image
					cv::circle(cv_depth_frame, fingerPoint, 4, cv::Scalar(255, 0, 0), 3);

					//draw finger point on color image
					cv::circle(cv_color_frame, fingerPoint, 4, cv::Scalar(255, 0, 0), 3);
				}




				if (!calibration_flag) {
					double timestamp = (double)(ms.count() - start_time.count()) - calibration_time;
					//outputFile << "Fingertip x = " << fingerPoint.x << ", y = " << fingerPoint.y << ". Time = " << timestamp << " ms." << std::endl;
					outputFile << "Fingertip x = " << fingerPoint.x << ", y = " << fingerPoint.y << ". Time = " << timestamp << " ms." << std::endl;
				}

			}
		}


		cv::imshow("depth", cv_depth_frame);

		if (calibration_flag) {
			putText(cv_color_frame, "Calibrating, please sit straight and don't move.", cvPoint(resolution_width / 10, resolution_height / 10), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(0, 255, 255), 1, CV_AA);
		}

		cv::imshow("color", cv_color_frame);
		
		
		if (waitKey(10) == 27)
		{
			break; // escape
		}


	}



	/*

	int camera_device = parser.get<int>("camera");
	VideoCapture capture;
	//-- 2. Read the video stream
	capture.open(camera_device);
	if (!capture.isOpened())
	{
	cout << "--(!)Error opening video capture\n";
	return -1;
	}

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

	Mat frame;
	while (capture.read(frame))
	{
	if (frame.empty())
	{
	cout << "--(!) No captured frame -- Break!\n";
	break;
	}
	//-- 3. Apply the classifier to the frame
	detectAndDisplay(frame);
	if (waitKey(10) == 27)
	{
	break; // escape
	}
	}


	*/


	return 0;
}