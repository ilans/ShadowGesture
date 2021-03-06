#pragma once
#include "opencv\cv.h"
#include "opencv\highgui.h"
//#include "opencv\ml.h"
#include <opencv2/gpu/gpu.hpp>

#include "boost/foreach.hpp"
#include "boost/regex.hpp"

//#include <boost/algorithm/string.hpp>
//#include <boost/lambda/lambda.hpp>
//#include "cvpr_stream.hpp"
//#include "array_utils.hpp"
//#include "pca.hpp"
//using namespace cvpr;
//using namespace cvpr::io;
//using namespace boost::lambda;

using namespace boost;

using namespace std;
using namespace cv;

class ShadowGesture
{
public:
	ShadowGesture(void);
	~ShadowGesture(void);

	struct mag_ang{
		float magnitude;
		float angle;
	};

	void capture(string path);
	void extract();
	void vectorize(string path);
	void trainHMM(string path);
	void testHMM(string path);
	static void on_mouse(int event, int x, int y, int flags, void* param);
	void loadImages(string dir_path, vector<string>& l);
	Mat getPointClusters(Mat& seqs, int num_frames, int num_observations, int num_features);
	Mat FindConvexityDefects(string path, Mat& image = Mat());
	mag_ang calcMagAng(Point2f& p);
	void recognizeGesture(string path);
	void convertDataToOctaveCVS(string path);
	void convertBinaryDataToOctaveCVS(string train_path, string test_path);
	void waitKeyPress();
	double prHmm(Mat& o);

	VideoCapture cap;

    vector<Point2f> video_corners;
	int video_width;
	int video_height;

    vector<Point2f> screen_corners;
	int screen_width;
	int screen_height;

	Mat frame;
	Mat screen;
	Mat screen_mat;

	bool mouse_down;
	int circle_radius;
	Point2f delta_mouse;
	int cur_corner;

	string v2s;

	float roi_height;

    vector<string> image_paths;

	Mat centers;
	PCA pca;
	int hidden_states;
	int num_hidden_states;
	int num_output_symbols;

	vector<Mat> hands_buf;
};


