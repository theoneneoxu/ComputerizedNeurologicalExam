#include <iostream>
#include <fstream>
#include <chrono>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <librealsense2/rs.hpp>

//int minH = 0, maxH = 20, minS = 30, maxS = 150, minV = 60, maxV = 255;
static int minH = 0, maxH = 17, minS = 255, maxS = 255, minV = 255, maxV = 255;
cv::Mat frame;
int count = 0;

static float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
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

void CallbackFunc(int event, int x, int y, int flags, void* userdata)
{
	cv::Mat RGB = frame(cv::Rect(x, y, 1, 1));
	cv::Mat HSV;
	cv::cvtColor(RGB, HSV, CV_BGR2HSV);
	cv::Vec3b pixel = HSV.at<cv::Vec3b>(0, 0);
	if (event == cv::EVENT_LBUTTONDBLCLK) // on double left clcik
	{
		std::cout << "Click" << std::endl;
		int h = pixel.val[0];
		int s = pixel.val[1];
		int v = pixel.val[2];
		if (count == 0)
		{
			minH = h;
			maxH = h;
			minS = s;
			maxS = s;
			minV = v;
			maxV = v;
		}
		else
		{
			if (h < minH)
			{
				minH = h;
			}
			else if (h > maxH)
			{
				maxH = h;
			}
			if (s < minS)
			{
				minS = s;
			}
			else if (s > maxS)
			{
				maxS = s;
			}
			if (v < minV)
			{
				minV = v;
			}
			else if (v > maxV)
			{
				maxV = v;
			}

		}
		count++;
	}
	std::cout << pixel << std::endl;
}

int mainaaaaa()
{

	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

	//color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2);
	color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0.f);	//fixed scale


	//Contruct a pipeline which abstracts the device
	rs2::pipeline pipe;

	//Create a configuration for configuring the pipeline with a non default profile
	rs2::config cfg;

	//Add desired streams to configuration
	cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
	cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);

	//Instruct pipeline to start streaming with the requested configuration
	pipe.start(cfg);

	// Camera warmup - dropping several first frames to let auto-exposure stabilize
	rs2::frameset frames;







	cv::VideoCapture cap(0);
	const char* windowName = "Fingertip detection";
	cv::namedWindow(windowName);



	std::cout << minH << std::endl;
	std::cout << maxH << std::endl;
	std::cout << minS << std::endl;
	std::cout << maxS << std::endl;
	std::cout << minV << std::endl;
	std::cout << maxV << std::endl;


	cv::setMouseCallback(windowName, CallbackFunc, NULL);


	std::cout << minH << std::endl;
	std::cout << maxH << std::endl;
	std::cout << minS << std::endl;
	std::cout << maxS << std::endl;
	std::cout << minV << std::endl;
	std::cout << maxV << std::endl;



	int inAngleMin = 200, inAngleMax = 300, angleMin = 180, angleMax = 359, lengthMin = 10, lengthMax = 80;
	cv::createTrackbar("Inner angle min", windowName, &inAngleMin, 360);
	cv::createTrackbar("Inner angle max", windowName, &inAngleMax, 360);
	cv::createTrackbar("Angle min", windowName, &angleMin, 360);
	cv::createTrackbar("Angle max", windowName, &angleMax, 360);
	cv::createTrackbar("Length min", windowName, &lengthMin, 100);
	cv::createTrackbar("Length max", windowName, &lengthMax, 100);



	cv::createTrackbar("MinH", windowName, &minH, 180);
	cv::createTrackbar("MaxH", windowName, &maxH, 180);
	cv::createTrackbar("MinS", windowName, &minS, 255);
	cv::createTrackbar("MaxS", windowName, &maxS, 255);
	cv::createTrackbar("MinV", windowName, &minV, 255);
	cv::createTrackbar("MaxV", windowName, &maxV, 255);


	std::ofstream outputFile;
	outputFile.open("Neo test finger tracking.txt");


	using namespace std::chrono;
	milliseconds start_time = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);




	while (1)
	{


		frames = pipe.wait_for_frames();
		//Get each frame
		rs2::frame color_frame = frames.get_color_frame();

		rs2::depth_frame depth = color_map(frames.get_depth_frame());
		//rs2::depth_frame depth = frames.get_depth_frame();

		// Query frame size (width and height)
		const int w = depth.get_width();
		const int h = depth.get_height();

		int memory_size = w * h * 3;
		unsigned char* test_data = new unsigned char[memory_size];
		std::fill_n(test_data, memory_size, 0);

		//std::cout << depth.get_distance(640, 360);
		
		/*
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int value = depth.get_distance(j, i) > 2 ? 0 : 255;
				int index = (i * w + j) * 3;
				test_data[index] = value;
			}
		}
		*/


		// Creating OpenCV Matrix from a color image
		//cv::Mat cvframe(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat cvframe(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
		frame = cvframe;

		//std::cout << frame;



		imshow("input", frame);




		//cap >> frame;
		cv::Mat hsv;
		cv::cvtColor(frame, hsv, CV_BGR2HSV);


		//imshow("hsv", hsv);



		cv::inRange(hsv, cv::Scalar(minH, minS, minV), cv::Scalar(maxH, maxS, maxV), hsv);

		
		imshow("after inrange", hsv);


		// Pre processing
		int blurSize = 5;
		int elementSize = 5;
		cv::medianBlur(hsv, hsv, blurSize);

		imshow("after median blur", hsv);

		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * elementSize + 1, 2 * elementSize + 1), cv::Point(elementSize, elementSize));
		cv::dilate(hsv, hsv, element);
		// Contour detection
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(hsv, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		size_t largestContour = 0;
		for (size_t i = 1; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > cv::contourArea(contours[largestContour]))
				largestContour = i;
		}
		cv::drawContours(frame, contours, largestContour, cv::Scalar(0, 0, 255), 1);


		//imshow("after contours", frame);


		// Convex hull
		if (!contours.empty())
		{
			std::vector<std::vector<cv::Point>> hull(1);
			cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
			cv::drawContours(frame, hull, 0, cv::Scalar(0, 255, 0), 3);
			if (hull[0].size() > 2)
			{

				/*
				std::vector<int> hullIndexes;
				cv::convexHull(cv::Mat(contours[largestContour]), hullIndexes, true);
				std::vector<cv::Vec4i> convexityDefects;
				cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndexes, convexityDefects);
				for (size_t i = 0; i < convexityDefects.size(); i++)
				{
					cv::Point p1 = contours[largestContour][convexityDefects[i][0]];
					cv::Point p2 = contours[largestContour][convexityDefects[i][1]];
					cv::Point p3 = contours[largestContour][convexityDefects[i][2]];
					cv::line(frame, p1, p3, cv::Scalar(255, 0, 0), 2);
					cv::line(frame, p3, p2, cv::Scalar(255, 0, 0), 2);
				}
				*/


				
				std::vector<int> hullIndexes;
				cv::convexHull(cv::Mat(contours[largestContour]), hullIndexes, true);
				std::vector<cv::Vec4i> convexityDefects;
				cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndexes, convexityDefects);
				cv::Rect boundingBox = cv::boundingRect(hull[0]);
				cv::rectangle(frame, boundingBox, cv::Scalar(255, 0, 0));
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
				for (size_t i = 0; i < validPoints.size(); i++)
				{
					//cv::circle(frame, validPoints[i], 9, cv::Scalar(0, 255, 0), 2);
				}
				
				cv::Point fingerPoint;
				fingerPoint.x = w;
				fingerPoint.y = h;
				for (size_t i = 0; i < validPoints.size(); i++)
				{
					if (validPoints[i].y < fingerPoint.y)
					{
						fingerPoint = validPoints[i];
					}
				}
				cv::circle(frame, fingerPoint, 4, cv::Scalar(255, 0, 0), 3);
				
				
				
				
				milliseconds ms = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				
				double timestamp = (double)(ms.count() - start_time.count()) / 1000;
				
				std::cout << "finger x = " << fingerPoint.x << ", y = " << fingerPoint.y << ". Time = " << timestamp << std::endl;




				outputFile << "finger x = " << fingerPoint.x << ", y = " << fingerPoint.y << ". Time = " << timestamp << std::endl;



			}
		}
		cv::imshow(windowName, frame);
		if (cv::waitKey(30) >= 0) break;



		delete[] test_data;


	}

	outputFile.close();

	return 0;
}