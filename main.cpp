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

//선
void connectLine(cv::Mat& img, full_object_detection landmarks, int iStart, int iEnd, bool isClosed = false)
{
	std::vector<cv::Point> points;
	for (int i = iStart; i < iEnd; i++)
	{
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
	}
	cv::polylines(img, points, isClosed, COLOR_DETECT, 2, 16);
}

// 랜드마크 그리기 함수
void drawPolygon(cv::Mat& img, full_object_detection landmarks)
{
	connectLine(img, landmarks, 0, 16); // 턱
	connectLine(img, landmarks, 17, 21); // 왼쪽 눈썹
	connectLine(img, landmarks, 22, 26); // 오른쪽 눈썹
	connectLine(img, landmarks, 27, 30); // 콧대
	connectLine(img, landmarks, 30, 35, true); // 낮은 코
	connectLine(img, landmarks, 36, 41, true); // 왼쪽 눈
	connectLine(img, landmarks, 42, 47, true); // 오른쪽 눈
	connectLine(img, landmarks, 48, 59, true); // 입술 바깥쪽
	connectLine(img, landmarks, 60, 67, true); // 입술 안쪽 부분
}

// dlib rectangle to opencv rect
cv::Rect dlibRectToOpencv(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
}

int main()
{
	Mat img = imread("users.jpg", IMREAD_COLOR); //객체생성
	if (img.empty()) //오류시 종료
	{
		cerr << "img open fail" << endl;
		return -1;
	}
	VideoCapture cap(1); //카메라
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
		drawPolygon(img, faceLandmark); //랜드마크 그리기
	}
	
	imshow("img", img); //영상출력

	Mat frame; //객체생성
	Mat sample1; //객체생성
	Mat sample2; //객체생성
	int x = 0; //객체생성
	int y = 0; //객체생성

	while (true) { //반복
		cap >> frame; //카메라
		if (frame.empty()) break; //오류시 break
		Mat blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123)); //객체생성
		net.setInput(blob);
		Mat res = net.forward(); //객체생성
		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>()); //객체생성
		for (int i = 0; i < detect.rows; i++) { //반복문
			float confidence = detect.at<float>(i, 2);
			if (confidence < 0.5) break;
			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols); //좌표
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows); //좌표
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols); //좌표
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows); //좌표
			circle(frame, Point((x1 + x2) / 2, (y1 + y2) / 2), ((y2 - y1) + (x2 - x1)) / 3, Scalar(0, 255, 0), 1); //얼굴 좌표에 원 그리기
			String label = format("user", confidence); //문자열
			putText(frame, label, Point(x1, y1 - 15), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0)); //글자출력
			cv_image<bgr_pixel> dlib_img(frame);
			std::vector<dlib::rectangle> faceRects = detector(dlib_img);
			int iFaceCount = faceRects.size();
			//imwrite("users.jpg", frame); //유저얼굴 저장시 사용
			for (int i = 0; i < iFaceCount; i++) //반복
			{
				full_object_detection faceLandmark = landmarkDetector(dlib_img, faceRects[i]);
				drawPolygon(frame, faceLandmark); //랜드마크 그리기
			}
			sample1 = frame(Rect(Point(x1 - 10, y1 - 10), Point(x2 + 10, y2 + 10))); //얼굴 크기 조정
			sample2 = img(Rect(Point(x1 - 10, y1 - 10), Point(x2 + 10, y2 + 10))); //얼굴 크기 조정

		}
		for (int i = 0; i < sample1.rows; i++) //반복
		{
			for (int j = 0; j < sample1.cols; j++) //반복
			{
				if (sample2.at<Vec3b>(j, i)[1] == 255) //sample2 색상이 (0,255,0)일 경우
				{
					y++; //증가
					if (sample1.at<Vec3b>(j, i)[1] == sample2.at<Vec3b>(j, i)[1]) //두 생상이 같을 경우
					{
						x++; //증가
					}
				}
			}
		}
		double e = (double)x / (double)y * 100.0; //인식률
		String ps = format("%4.3f", e); //문자열
		putText(frame, ps, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0)); //글자출력
		imshow("sample1", sample1); //영상출력
		imshow("sample2", sample2); //영상출력
		imshow("frame", frame); //영상출력
		if (e>70.0) break; //인식률70%이상일 경우 break
		else //70이하일 경우 계수 초기화
		{
			x = 0; //대입
			y = 0; //대입
		}
		if (waitKey(1) == 27) break; //esc입력시 break
	}
	
	destroyAllWindows();
	return 0; //종료
}