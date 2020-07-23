#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace cv;
using namespace std;
using namespace xfeatures2d;






void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void printMat(Mat mat);
Point p1;
Point p2;

bool clickedD = false;
bool started = true;
int main()
{

	VideoCapture videoCapture(0);
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 1200);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	Mat img;
	Rect2d mouse_rect;
	Mat obj;
	Mat prev_img_gray;
	vector<Point2f> prev_pts;
	Mat image_out;
	bool matchFeature = false;
	namedWindow("mainImg", 1);
	setMouseCallback("mainImg", CallBackFunc, NULL);
	bool detectionAppplied = false;

	int minHessian = 400;
	int minKeypoints = 40;
	while (true)
	{


		videoCapture >> img;
		if (img.empty())
		{
			cout << "Empty image\n \n";
			break;
		}

		string message = "";
		if (clickedD&&started)
		{
			message = "Druecken Sie \"c\", um den gesamten Vorgang abzubrechen oder Druecken Sie \"a\" um das Tracking zu starten";
			if (p1.x > p2.x) {
				mouse_rect.x = p2.x;
				mouse_rect.width = p1.x - p2.x;
			}
			else {
				mouse_rect.x = p1.x;
				mouse_rect.width = p2.x - p1.x;
			}

			if (p1.y > p2.y) {
				mouse_rect.y = p2.y;
				mouse_rect.height = p1.y - p2.y;
			}
			else {
				mouse_rect.y = p1.y;
				mouse_rect.height = p2.y - p1.y;
			}

			rectangle(img, mouse_rect, Scalar(255, 255, 0));


		}
		else 	message = "Selektieren Sie den Bereich oder Druecken Sie \"c\", um den gesamten Vorgang abzubrechen.";

		putText(img, message, cvPoint(0, img.rows - 10),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
		imshow("mainImg", img);

		if (mouse_rect.area() > 0 && matchFeature)
		{
			Mat img_gray = img.clone();
			Mat mask = Mat::zeros(img_gray.size(), CV_8UC1);
			rectangle(mask, mouse_rect, Scalar::all(255), -1);


			vector<Point2f> pts;
			if (!detectionAppplied)
			{
				Ptr<Feature2D> detector = SURF::create(minHessian);
				vector<KeyPoint> keypoints;

				detector->detect(img_gray, keypoints, mask);
				cout << "found " << keypoints.size() << " keypoints" << endl;
				if (keypoints.size() < minKeypoints)
				{
			
					continue;
				}

				for (int i = 0; i < keypoints.size(); ++i)
					pts.push_back(keypoints[i].pt);

				prev_img_gray = img_gray.clone();
				prev_pts = pts;
				detectionAppplied = true;
				continue;

			}
			 image_out = img_gray.clone();




			vector<unsigned char> status;
			vector<float> err;




			calcOpticalFlowPyrLK(prev_img_gray, img_gray, prev_pts, pts, status, err);

			for (int i = 0; i < pts.size(); ++i)
				if (status[i] != 0)
					circle(image_out, pts[i], 3, Scalar(255, 0, 0));

			for (int i = 0; i < prev_pts.size(); i++)
					if (status[i] != 0)
				circle(image_out, prev_pts[i], 3, Scalar(0, 255, 0));

			for (int i = 0; i < pts.size(); i++)
				if (status[i] != 0)
					line(image_out, prev_pts[i], pts[i], Scalar(0, 255, 255));

			imshow("Output", image_out);



			prev_img_gray = img_gray.clone();

			int i = 0;
			prev_pts.clear();
		
				for (int i = 0; i < pts.size(); i++)
				{
					if (status[i] != 0)
						prev_pts.push_back(pts[i]);
				}
				if (prev_pts.size()< minKeypoints)
					detectionAppplied = false;
				
		






		}



		int c = waitKey(10);
		if (c == 'c')
			break;
		else if (c == 'a') {
			matchFeature = !matchFeature;
		}

	}
	return 0;
}



void printMat(Mat mat)
{
	for (int i = 0; i < mat.size().height; i++)
	{
		cout << "[";
		for (int j = 0; j < mat.size().width; j++)
		{
			cout << std::fixed << setw(10) << setprecision(2) << mat.at<double>(i, j);
			if (j != mat.size().width - 1)
				cout << ", ";
			else
				cout << "]" << endl;
		}
	}
}


//Callback Methode der Maus
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		//Clicked und nicht losgelassen

		clickedD = true;
		started = false;
		p1.x = x;
		p1.y = y;

	}

	else if (event == EVENT_MOUSEMOVE)
	{
		if (clickedD)
		{
			//clicked, nicht losgelassen und bewegt
			p2.x = x;
			p2.y = y;
			started = true;
		}

	}
	else if (event == EVENT_LBUTTONUP)
	{
		if (clickedD)
		{
			//clicked und wird losgelassen
			p2.x = x;
			p2.y = y;
			started = false;
			clickedD = false;
		}

	}
}