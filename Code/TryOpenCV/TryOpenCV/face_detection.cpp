#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		//"{face_cascade|haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{face_cascade|lbpcascade_frontalface.xml|Path to face cascade.}"
		//"{face_cascade|haarcascade_mcs_nose.xml|Path to face cascade.}"
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
	return 0;
}
void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;


	Mat small_frame;
	frame_gray;


	Rect region_of_interest = Rect(160, 90, 360, 240);
	Mat roi = frame_gray(region_of_interest);


	//face_cascade.detectMultiScale(frame_gray, faces);
	//face_cascade.detectMultiScale(frame_gray, faces, 1.2, 3, 0, cv::Size(20, 20), cv::Size(200, 200));

	//test for nose
	//too slow to directly detect nose
	//face_cascade.detectMultiScale(roi, faces, 1.1, 3);

	//for face detection
	face_cascade.detectMultiScale(roi, faces, 1.25, 3, 0, cv::Size(50, 50), cv::Size(200, 200));




	for (size_t i = 0; i < faces.size(); i++)
	{
		//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		Point center(faces[i].x + faces[i].width / 2 + 160, faces[i].y + faces[i].height / 2 + 90);
		
		
		circle(frame, center, 2, Scalar(255, 0, 255), 4);

		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);



		int x = faces[i].x + 160;
		int y = faces[i].y + 90;
		int width = faces[i].width;
		int height = faces[i].height;

		Rect noseRect = Rect(x, y, width, height);
		cv::Point pt1(x, y);
		cv::Point pt2(x + width, y + height);
		//cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0));

		cv::Point pt3(faces[i].x, faces[i].y);
		cv::Point pt4(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		//cv::rectangle(frame, pt3, pt4, cv::Scalar(255, 255, 0));

		//Mat faceROI = frame_gray(faces[i]);
		Mat faceROI = frame_gray(noseRect);



		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		
		
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3);
		
		
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2 + 160, faces[i].y + eyes[j].y + eyes[j].height / 2 + 90);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.2);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}
		

	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}