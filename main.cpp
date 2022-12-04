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

//��
void connectLine(cv::Mat& img, full_object_detection landmarks, int iStart, int iEnd, bool isClosed = false)
{
	std::vector<cv::Point> points;
	for (int i = iStart; i < iEnd; i++)
	{
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
	}
	cv::polylines(img, points, isClosed, COLOR_DETECT, 2, 16);
}

// ���帶ũ �׸��� �Լ�
void drawPolygon(cv::Mat& img, full_object_detection landmarks)
{
	connectLine(img, landmarks, 0, 16); // ��
	connectLine(img, landmarks, 17, 21); // ���� ����
	connectLine(img, landmarks, 22, 26); // ������ ����
	connectLine(img, landmarks, 27, 30); // ���
	connectLine(img, landmarks, 30, 35, true); // ���� ��
	connectLine(img, landmarks, 36, 41, true); // ���� ��
	connectLine(img, landmarks, 42, 47, true); // ������ ��
	connectLine(img, landmarks, 48, 59, true); // �Լ� �ٱ���
	connectLine(img, landmarks, 60, 67, true); // �Լ� ���� �κ�
}

// dlib rectangle to opencv rect
cv::Rect dlibRectToOpencv(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
}

int main()
{
	Mat img = imread("users.jpg", IMREAD_COLOR); //��ü����
	if (img.empty()) //������ ����
	{
		cerr << "img open fail" << endl;
		return -1;
	}
	VideoCapture cap(1); //ī�޶�
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
		drawPolygon(img, faceLandmark); //���帶ũ �׸���
	}
	
	imshow("img", img); //�������

	Mat frame; //��ü����
	Mat sample1; //��ü����
	Mat sample2; //��ü����
	int x = 0; //��ü����
	int y = 0; //��ü����

	while (true) { //�ݺ�
		cap >> frame; //ī�޶�
		if (frame.empty()) break; //������ break
		Mat blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123)); //��ü����
		net.setInput(blob);
		Mat res = net.forward(); //��ü����
		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>()); //��ü����
		for (int i = 0; i < detect.rows; i++) { //�ݺ���
			float confidence = detect.at<float>(i, 2);
			if (confidence < 0.5) break;
			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols); //��ǥ
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows); //��ǥ
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols); //��ǥ
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows); //��ǥ
			circle(frame, Point((x1 + x2) / 2, (y1 + y2) / 2), ((y2 - y1) + (x2 - x1)) / 3, Scalar(0, 255, 0), 1); //�� ��ǥ�� �� �׸���
			String label = format("user", confidence); //���ڿ�
			putText(frame, label, Point(x1, y1 - 15), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0)); //�������
			cv_image<bgr_pixel> dlib_img(frame);
			std::vector<dlib::rectangle> faceRects = detector(dlib_img);
			int iFaceCount = faceRects.size();
			//imwrite("users.jpg", frame); //������ ����� ���
			for (int i = 0; i < iFaceCount; i++) //�ݺ�
			{
				full_object_detection faceLandmark = landmarkDetector(dlib_img, faceRects[i]);
				drawPolygon(frame, faceLandmark); //���帶ũ �׸���
			}
			sample1 = frame(Rect(Point(x1 - 10, y1 - 10), Point(x2 + 10, y2 + 10))); //�� ũ�� ����
			sample2 = img(Rect(Point(x1 - 10, y1 - 10), Point(x2 + 10, y2 + 10))); //�� ũ�� ����

		}
		for (int i = 0; i < sample1.rows; i++) //�ݺ�
		{
			for (int j = 0; j < sample1.cols; j++) //�ݺ�
			{
				if (sample2.at<Vec3b>(j, i)[1] == 255) //sample2 ������ (0,255,0)�� ���
				{
					y++; //����
					if (sample1.at<Vec3b>(j, i)[1] == sample2.at<Vec3b>(j, i)[1]) //�� ������ ���� ���
					{
						x++; //����
					}
				}
			}
		}
		double e = (double)x / (double)y * 100.0; //�νķ�
		String ps = format("%4.3f", e); //���ڿ�
		putText(frame, ps, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0)); //�������
		imshow("sample1", sample1); //�������
		imshow("sample2", sample2); //�������
		imshow("frame", frame); //�������
		if (e>70.0) break; //�νķ�70%�̻��� ��� break
		else //70������ ��� ��� �ʱ�ȭ
		{
			x = 0; //����
			y = 0; //����
		}
		if (waitKey(1) == 27) break; //esc�Է½� break
	}
	
	destroyAllWindows();
	return 0; //����
}