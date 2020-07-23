#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace cv;
using namespace std;
using namespace xfeatures2d;

//PATH=C:\OpenCV3.4.0\x64\vc14\bin;C:\OpenCV3.4.0\tbb\bin\intel64\vc14




void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void printMat(Mat mat);

//Erster Punkt des selektierten Rechtecks
Point p1;
//Zweiter Punkt des selektierten Rechtecks
Point p2;

//Bestimmt ob die linke Maustaste gedrückt und nich losgelassen wurde
bool clickedD = false;
//Bestimmt ob die linke Maustaste gedrückt und nicht losgelassen und zum Zeichnen, sich bewegt hat
bool started = true;
int main()
{

	VideoCapture videoCapture(0);

	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 1200);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	//Ursprungbild
	Mat img;

	//Selektiertes Rechteck
	Rect mouse_rect;

	//Selektiertes Objekt-Bild aus dem Ursprungsbild
	Mat obj;
	namedWindow("mainImg", 1);

	//Setzzt die Callback-Methode bei irgendeinem Ereignis von der Maus 
	setMouseCallback("mainImg", CallBackFunc, NULL);

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
			message = "Druecken Sie \"c\", um den gesamten Vorgang abzubrechen oder Druecken Sie \"s\" um den Bereich definitiv zu selektieren";

			// Zeichnet das Rechteck
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
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
		//	resizeWindow("mainImg", 700, 500);
		//Zeigt das neue Bild mit dem gezeichneten Rechteck drauf
		imshow("mainImg", img);

		if (!obj.empty())
		{
			int minHessian = 400;
			Ptr<Feature2D> detector = SURF::create(minHessian);
			vector<KeyPoint> obj_keypoints; // Keypunkte des selektierten Rechteck-Bildes
			Mat obj_descriptors; // Deskriptoren des selektierten Rechteck-Bildes

			detector->detectAndCompute(obj, noArray(), obj_keypoints, obj_descriptors);

			cout << "\n" << obj_keypoints.size() << " Objekt-Merkmalspunkte gefunden.\n";
			if (obj_keypoints.size() >= 20)
			{



				vector<KeyPoint> scene_keypoints; // Merkmalspunkte der Szene
				Mat scene_descriptors; // Merkmalsvektoren der Szene
				detector->detectAndCompute(img, noArray(), scene_keypoints, scene_descriptors);

				cout << "\n" << scene_keypoints.size() << " Szene-Merkmalspunkte gefunden.\n";

				FlannBasedMatcher matcher;

				vector<vector<DMatch>> matches;

				matcher.knnMatch(scene_descriptors, obj_descriptors, matches, 2); // finde die 2 nahesten Nachbarn 

				vector<DMatch> good_matches;
				for (auto match : matches)
				{
					if (match[0].distance < 0.75*(match[1].distance))
						good_matches.push_back(match[0]);
				}

				Mat img_gray = img.clone();
				cvtColor(img_gray, img_gray, cv::COLOR_BGR2GRAY);
				//img_gray = Scalar(255, 255, 255) - img_gray;
				Mat image_out;
				drawMatches(img_gray, scene_keypoints, obj, obj_keypoints, good_matches, image_out);



				vector<Point2f> objPts;
				vector<Point2f> scenePts;



				for (unsigned int i = 0; i < good_matches.size(); i++)
				{
					//Get the keypoints from the good matches
					objPts.push_back(obj_keypoints[good_matches[i].trainIdx].pt);
					scenePts.push_back(scene_keypoints[good_matches[i].queryIdx].pt);
				}



				//Herausfinden der Homographiematrix
				Mat H = findHomography(objPts, scenePts, cv::RANSAC);
				if (!H.empty())
				{
					cout << "\H ist : \n";
					printMat(H);
					vector<Point2f> obj_corners(4);
					obj_corners[0] = Point(0, 0);
					obj_corners[1] = Point(obj.cols, 0);
					obj_corners[2] = Point(obj.cols, obj.rows);
					obj_corners[3] = Point(0, obj.rows);

					vector<Point2f> scene_corners(4);

					perspectiveTransform(obj_corners, scene_corners, H);

					Point2f p0 = Point2f(obj.cols, 0);

					line(image_out, p0 + scene_corners[0], p0 + scene_corners[1], Scalar(0, 255, 0), 3);
					line(image_out, p0 + scene_corners[1], p0 + scene_corners[2], Scalar(0, 255, 0), 3);
					line(image_out, p0 + scene_corners[2], p0 + scene_corners[3], Scalar(0, 255, 0), 3);
					line(image_out, p0 + scene_corners[3], p0 + scene_corners[0], Scalar(0, 255, 0), 3);

				}


				//Anzeigen des Gesamtbildes
				imshow("Output", image_out);
			}
		}



		int c = waitKey(10);
		if (c == 'c')
			break;
		//
		else if (c == 's'&&clickedD&&started)
		{

			Mat tmp_img(img, mouse_rect);
			tmp_img.copyTo(obj);

			clickedD = false;
			started = false;//BGR2GRAY
			cvtColor(obj, obj, cv::COLOR_BGR2GRAY);
			imshow("SelectedImg", obj);

		}
	}
	return 0;
}


//ordentliche Anzeige einer Matrix
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