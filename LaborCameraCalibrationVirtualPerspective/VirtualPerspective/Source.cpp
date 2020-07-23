
#include <opencv2\opencv.hpp>
#include <opencv2\aruco.hpp>
#include <iostream>
#include <list>
#include <cmath>
#include <thread>

using namespace cv;
using namespace std;



int main()

{


		VideoCapture videoCapture(0);
		videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 2000);
		videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	FileStorage fs("J:/Bauernöppel/M8 Bild- und Videoverarbeitung/lifecam.yml", FileStorage::READ);

	Mat K; // Für die Kamera-Matrix
	Mat d; //Distorsion Matrix
	fs["camera_matrix"] >> K;
	fs["distortion_coefficients"] >> d;
	bool detectCorners = false;
	bool detectAcuro = false;
	string message = "Druecken Sie \"c\" fuer die Eckpunkte-Detktierung  und \"a\" fuer Aruco-Detektierung. Zum Abbrechen des Vorgangs druecken Sie \"x\"";
	//float pitch = 0;
	while (1)
	{
		Mat img;
		videoCapture >> img;
		if (img.empty())
		{
			cout << "End of video\n";
			break;
		}

		putText(img, message, cvPoint(0, img.rows - 10),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
		imshow("Ursprung Bild", img);

		if (detectCorners)
		{
			Mat out = img.clone();

			vector<Point2f> corners;
			int boardW = 9;
			int boardH = 6;


			bool found = findChessboardCorners(out, Size(boardW, boardH), corners, CALIB_CB_ADAPTIVE_THRESH);
			if (found)
			{

				out = Scalar(255, 255, 255) - out;
				Point2f imgPts[4];
				imgPts[0] = corners[0];
				imgPts[1] = corners[boardW - 1];
				imgPts[2] = corners[(boardH - 1)*boardW];
				imgPts[3] = corners[(boardH - 1)*boardW + (boardW - 1)];

				circle(out, imgPts[0], 9, Scalar(255, 0, 0), 3);
				circle(out, imgPts[1], 9, Scalar(0, 255, 0), 3);
				circle(out, imgPts[2], 9, Scalar(0, 0, 255), 3);
				circle(out, imgPts[3], 9, Scalar(255, 255, 0), 3);

				imshow("Neues Bild", out);

				float length = 20; 
				float X = length * (boardW - 1); 
				float Y = length * (boardH - 1); 

				Point2f objPts[4];
				objPts[0].x = 0; objPts[0].y = 0;
				objPts[1].x = X; objPts[1].y = 0;
				objPts[2].x = 0; objPts[2].y = Y;
				objPts[3].x = X; objPts[3].y = Y;

				Mat H = getPerspectiveTransform(objPts, imgPts);
				cout << "\nhomography matrix: \n" << H;

				Mat birdView;

				warpPerspective(out, birdView, H, img.size(), WARP_INVERSE_MAP + INTER_LINEAR, BORDER_CONSTANT, Scalar::all(0));
				imshow("bird", birdView);


				vector<Point2f> image_points;
				vector<Point3f> object_points;

				image_points.push_back(imgPts[0]);
				image_points.push_back(imgPts[1]);
				image_points.push_back(imgPts[2]);
				image_points.push_back(imgPts[3]);


				for (Point2f elem : objPts)
					object_points.push_back(Point3f(elem.x, elem.y, 0));

				Mat rvec, tvec;

				bool 	solved = solvePnP(object_points, image_points, K, d, rvec, tvec);

				double rotationAngle = norm(rvec);
				cout << "rvec: " << rvec << "\n";
				cout << "tvec: " << tvec << "\n";

				cout << "Rotation angle in grad: " << rotationAngle << "rad = " << (rotationAngle*180.0) / CV_PI << "degree \n\n";



			}

		}

		if (detectAcuro)
		{
			Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

			Ptr<aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);
			Mat boardImage;
			board->draw(cv::Size(2100, 2800), boardImage, 10, 1);
			//imshow("boardimg", boardImage);
			//imwrite("aruco.png", boardImage);



			vector<vector<Point2f>> corners;
			vector<int> ids;
			aruco::detectMarkers(img, dictionary, corners, ids);


			if (ids.size() > 0)
			{
				aruco::drawDetectedMarkers(img, corners, ids);


				vector<Vec3d> rvecs;
				vector<Vec3d> tvecs;
				aruco::estimatePoseSingleMarkers(corners, 1.0, K, d, rvecs, tvecs);
				Mat out = img.clone();
				for (int i = 0; i<ids.size(); i++)
				{
					int id = ids[i];
					aruco::drawAxis(out, K, d, rvecs[i], tvecs[i], 0.5f);
				}
				imshow("Axenbild", out);
			}
		

		}





		/*	Picts.push_front(clone);*/
		int c = waitKey(10);

		if (c == 'x') break;
		if (c == 'c') detectCorners = !detectCorners;
		if (c == 'a') detectAcuro = !detectAcuro;





	}

	return 0;
}


