#include <iostream>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace dlib;

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";

#define COLOR_DETECT Scalar(0, 255, 0)

//¼±
void connectLine(cv::Mat& img, full_object_detection landmarks, int iStart, int iEnd, bool isClosed = false)
{
	std::vector<cv::Point> points;
	for (int i = iStart; i < iEnd; i++)
	{
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
	}
	cv::polylines(img, points, isClosed, COLOR_DETECT, 2, 16);
}

// ·£µå¸¶Å© ±×¸®±â ÇÔ¼ö
void drawPolygon(cv::Mat& img, full_object_detection landmarks)
{
	connectLine(img, landmarks, 0, 16); // ÅÎ
	connectLine(img, landmarks, 17, 21); // ¿ÞÂÊ ´«½ç
	connectLine(img, landmarks, 22, 26); // ¿À¸¥ÂÊ ´«½ç
	connectLine(img, landmarks, 27, 30); // Äà´ë
	connectLine(img, landmarks, 30, 35, true); // ³·Àº ÄÚ
	connectLine(img, landmarks, 36, 41, true); // ¿ÞÂÊ ´«
	connectLine(img, landmarks, 42, 47, true); // ¿À¸¥ÂÊ ´«
	connectLine(img, landmarks, 48, 59, true); // ÀÔ¼ú ¹Ù±ùÂÊ
	connectLine(img, landmarks, 60, 67, true); // ÀÔ¼ú ¾ÈÂÊ ºÎºÐ
}

// dlib rectangle to opencv rect
cv::Rect dlibRectToOpencv(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
}

int main()
{
	Mat img = imread("users.jpg", IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "img open fail" << endl;
		return -1;
	}
	VideoCapture cap(1);
	if (!cap.isOpened()) { cerr << "Camera open failed!" << endl; return -1; }
	Net net = readNet(model, config);
	if (net.empty()) { cerr << "Net open failed!" << endl; return -1; }
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor landmarkDetector;
	deserialize(".\\shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

	cv_image<bgr_pixel> dlib_img(img);
	std::vector<dlib::rectangle> faceRects = detector(dlib_img);
	int iFaceCount = faceRects.size();

	// draw
	for (int i = 0; i < iFaceCount; i++)
	{
		full_object_detection faceLandmark = landmarkDetector(dlib_img, faceRects[i]);
		drawPolygon(img, faceLandmark);
	}
	
	imshow("img", img);

	Mat frame;
	Mat sample1;
	Mat sample2;
	int x = 0;
	int y = 0;

	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		Mat blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123));
		net.setInput(blob);
		Mat res = net.forward();
		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>());
		for (int i = 0; i < detect.rows; i++) {
			float confidence = detect.at<float>(i, 2);
			if (confidence < 0.5) break;
			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols);
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows);
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols);
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows);
			circle(frame, Point((x1 + x2) / 2, (y1 + y2) / 2), ((y2 - y1) + (x2 - x1)) / 3, Scalar(0, 255, 0), 1);
			String label = format("user", confidence);
			putText(frame, label, Point(x1, y1 - 15), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0));
			cv_image<bgr_pixel> dlib_img(frame);
			std::vector<dlib::rectangle> faceRects = detector(dlib_img);
			int iFaceCount = faceRects.size();
			//imwrite("users.jpg", frame);
			// draw
			for (int i = 0; i < iFaceCount; i++)
			{
				full_object_detection faceLandmark = landmarkDetector(dlib_img, faceRects[i]);
				drawPolygon(frame, faceLandmark);
			}
			sample1 = frame(Rect(Point(x1 - 10, y1 - 10), Point(x2 + 10, y2 + 10)));
			sample2 = img(Rect(Point(x1 - 10, y1 - 10), Point(x2 + 10, y2 + 10)));

		}
		for (int i = 0; i < sample1.rows; i++)
		{
			for (int j = 0; j < sample1.cols; j++)
			{
				if (sample2.at<Vec3b>(j, i)[1] == 255)
				{
					y++;
					if (sample1.at<Vec3b>(j, i)[1] == sample2.at<Vec3b>(j, i)[1])
					{
						x++;
					}
				}
			}
		}
		double e = (double)x / (double)y * 100.0;
		String ps = format("%4.3f", e);
		putText(frame, ps, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0));
		imshow("sample1", sample1);
		imshow("sample2", sample2);
		imshow("frame", frame);
		if (e>70.0) break;
		else
		{
			x = 0;
			y = 0;
		}
		if (waitKey(1) == 27) break;
	}
	
	destroyAllWindows();
	return 0;
}