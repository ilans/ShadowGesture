#include "StdAfx.h"
#include "ShadowGesture.h"
#include <fstream>
#include <dirent.h>
#include "CvHMM.h"

#define PI 3.14159

ShadowGesture::ShadowGesture(void)
{
}


ShadowGesture::~ShadowGesture(void)
{
}

void ShadowGesture::capture(){
	cout << "Staring ShadowGesture..." << endl;

	cap.open("../Data/gestures_for_shadowmonster2.mov");
	namedWindow("video");

	video_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	video_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	screen_width = 1280/2;
	screen_height = 720/2;

	v2s = "../Data/video2screen.xml";
	FileStorage fs(v2s, FileStorage::READ);
	if( fs.isOpened() )
	{
		fs["V"] >> video_corners;
		fs["S"] >> screen_corners;

		fs.release();
	}
	else
	{
		fs.release();
		FileStorage fs(v2s, FileStorage::WRITE);

		video_corners.push_back( Point2f(video_width*2/6, video_height*2/6) );//UL
		video_corners.push_back( Point2f(video_width*4/6, video_height*2/6) );//UR
		video_corners.push_back( Point2f(video_width*4/6, video_height*4/6) );//LR
		video_corners.push_back( Point2f(video_width*2/6, video_height*4/6) );//LL

		screen_corners.push_back( Point2f(0, 0) );//UL
		screen_corners.push_back( Point2f(screen_width, 0) );//UR
		screen_corners.push_back( Point2f(screen_width, screen_height) );//LR
		screen_corners.push_back( Point2f(0, screen_height) );//LL

		fs << "V" << video_corners;
		fs << "S" << screen_corners;

		fs.release();
	}

	mouse_down = false;
	circle_radius = 6;
	roi_height = 100;
	screen = Mat(screen_height, screen_width, CV_8UC1);

	screen_mat = findHomography(video_corners, screen_corners);
	setMouseCallback("video", (MouseCallback)on_mouse, this);

	double frame_time = 1/cap.get(CV_CAP_PROP_FRAME_COUNT);
	bool play_all = false;
	bool play_one = true;
	bool done = false;
	for(;;){
		if(play_one){
			cap >> frame;
			play_one = false;
		} else if(play_all){
			cap >> frame;
		}

		if(!frame.data) break;

		warpPerspective(frame, screen, screen_mat, screen.size());
		cvtColor(screen, screen, CV_RGB2GRAY);
		threshold(screen, screen, 100, 255, THRESH_BINARY_INV);

		Mat screen_c = screen.clone();
		Mat screen_cc = screen.clone();
		Mat screen_hand = screen.clone();

		Mat screen_rgb;
		cvtColor(screen, screen_rgb, CV_GRAY2BGR);

		// find touch points
		vector< vector<Point2i> > contours;
		vector<Point2f> touchPoints;
		vector<Mat> handRects;
		findContours(screen_c, contours, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE );
		//drawContours(screen_rgb, contours, -1, Scalar(0,0,255), 2);

		for (unsigned int i=0; i<contours.size(); i++) {
			Mat contourMat(contours[i]);
			Rect brect = boundingRect(contourMat);
			if ( contourArea(contourMat) > 3000 ) {
				Rect urect(brect.x, brect.y, brect.width, roi_height);
				rectangle(screen_rgb, urect, Scalar(0,255,255));
				if(urect.y<screen_height-roi_height){
					vector< vector<Point2i> > icontours;
					findContours(screen_cc(urect), icontours, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE, brect.tl());
					for (unsigned int i=0; i<icontours.size(); i++) {
						Mat iContourMat(icontours[i]);
						Rect hrect = boundingRect(iContourMat);
						Rect frect;
						if(hrect.width<roi_height){
							Point p(hrect.x+hrect.width/2-roi_height/2, hrect.y);
							frect = Rect(p.x, p.y, roi_height, roi_height);
						} else if(hrect.x == urect.x+1){
							frect = Rect(hrect.x, hrect.y, roi_height, roi_height);
						} else {
							//stringstream s;
							//s << hrect.x-urect.x;
							//putText(screen_rgb, s.str(), Point(100,100), 1, 1, Scalar(0,0,255));
							frect = Rect(urect.x+urect.width-roi_height, urect.y, roi_height, roi_height);
						}
						rectangle(screen_rgb, frect, Scalar(0,0,255));
						if(frect.y<screen_height-roi_height && frect.x<screen_width-roi_height && frect.y>0 && frect.x>0){
							handRects.push_back(screen_hand(frect));
						}
					}
				}
			}
		}

		for (int i=0 ; i<handRects.size() ; ++i) {
			stringstream s;
			s << i;
			stringstream f;
			f << cap.get(CV_CAP_PROP_POS_FRAMES);
			imshow("hand"+s.str(), handRects[i]);
			imwrite("../Data/images/hand"+f.str()+"_"+s.str()+".png", handRects[i]);
		}


		imshow("screen", screen_rgb);

		for (int i=0 ; i<video_corners.size() ; ++i) {
			circle(frame, video_corners[i] , circle_radius, Scalar(0,0,255), 1);
			if(i<screen_corners.size()-1)
				line(frame, video_corners[i], video_corners[i+1], Scalar(255,0,0));
			else
				line(frame, video_corners[i], video_corners[0], Scalar(255,0,0));
		}
		imshow("video", frame);

		int key = waitKey(1);
		double cur_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
		double cur_time = 0;
		if(key >= 0){
			switch(key){
				case 2424832://LEFT
					break;
				case 2555904://RIGHT
					cur_time = (cur_frame+10.0)*frame_time-1;
					cap.set(CV_CAP_PROP_POS_AVI_RATIO, frame_time*50-1);
					printf("cur_time: %f\n", frame_time);
					play_one = true;
					break;
				case 2490368://UP
					break;
				case 2621440://DOWN
					break;
				case 32://SPACE
					play_all = !play_all;
					break;
				case 27://ESC
					done = true;
					break;
			}
		}

		if(done)
			break;
	}
}

void ShadowGesture::on_mouse(int event, int x, int y, int flags, void* param)
{
	ShadowGesture* obj = (ShadowGesture*)param;

	Point2f point(x,y);
	if( !obj->mouse_down && event == CV_EVENT_LBUTTONDOWN)
	{
		for (int i=0 ; i<obj->video_corners.size() ; ++i) {
			Point2f corner = obj->video_corners[i];
			float rect_x = corner.x - obj->circle_radius;
			float rect_y = corner.y - obj->circle_radius;
			float side = obj->circle_radius*2;
			Rect rect = Rect(rect_x, rect_y, side, side);
			if(point.inside(rect)){
				obj->delta_mouse = point - corner;
				obj->cur_corner = i;
				obj->mouse_down = true;
				break;
			}
		}
	}
	else if( obj->mouse_down && event == CV_EVENT_MOUSEMOVE)
	{
		obj->video_corners[obj->cur_corner] = point - obj->delta_mouse;
		obj->screen_mat = findHomography(obj->video_corners, obj->screen_corners);
	}
	else if( obj->mouse_down && event == CV_EVENT_LBUTTONUP)
	{
		FileStorage fs(obj->v2s, FileStorage::WRITE);
		fs << "V" << obj->video_corners;
		fs << "S" << obj->screen_corners;
		fs.release();
		obj->mouse_down = false;
	}
}

bool sortByNumber(string a, string b) {
	int an = atoi(a.substr(19,a.size()-6).c_str());
	int bn = atoi(b.substr(19,b.size()-6).c_str());
	return (an<bn); 
}

void ShadowGesture::extract(){
	cout << "Staring extraction..." << endl;

	loadImages("../Data/images", image_paths);
	sort(image_paths.begin(), image_paths.end(), sortByNumber);

	bool done = false;
	bool next = false;
	bool prev = false;
	int i = 0;

	Mat image;
	string path;

	vector< vector<string> > seqs;
	vector<string> seq;

	FileStorage fs("../Data/sequense.yml", FileStorage::WRITE);
	int seq_i = 1;
	for(;;){
		if(next){
			if(i<image_paths.size())
				++i;
			next = false;
		} else if(prev){
			if(i>0)
				--i;
			prev = false;
		}

		string path = image_paths[i];
		image = imread(path);

		string num = path.substr(19,image_paths[i].size()-23);
		rectangle(image, Point(20,85), Point(80,95), Scalar(255,255,255), -1);
		putText(image, num, Point(22,94), CV_FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,100));

		vector<string>::iterator it;
		it = find(seq.begin(), seq.end(), path);
		if(it != seq.end()){
			rectangle(image, Point(0,0), Point(100,100), Scalar(0,0,255), 3);
		}

		imshow("image", image);

		it = find(seq.begin(), seq.end(), path);

		int seq_i = 0;

		int key = waitKey(1);
		if(key >= 0){
			switch(key){
				case 2424832://LEFT
					prev = true;
					break;
				case 2555904://RIGHT
					next = true;
					break;
				case 2490368://UP
					if(it == seq.end()){
						seq.push_back(path);
					}
					break;
				case 2621440://DOWN
					if(it != seq.end()){
						seq.erase(it);
					}
					break;
				case 3014656://DELETE
					if(it != seq.end()){
						seq.erase(it);
					}
					image_paths.erase(find(image_paths.begin(), image_paths.end(), path));
					remove(path.c_str());
					break;
				case 32://SPACE
					sort(seq.begin(), seq.end(), sortByNumber);
					seqs.push_back(seq);
					seq.clear();
					break;
				case 27://ESC
					done = true;
					break;
			}
		}

		if(done){
			fs << "seqs" << "[";
			BOOST_FOREACH(vector<string>& seq, seqs){
				fs << seq;
			}
			fs << "]";
			fs.release();
			break;
		}
	}
}

void ShadowGesture::loadImages(string dir_path, vector<string>& l){
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir_path.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir_path << endl;
        return;
    }
    
    while ((dirp = readdir(dp)) != NULL) {
        string file_path = dir_path+"/"+string(dirp->d_name);
        
        smatch what;
        if( !regex_match( file_path, what, boost::regex(".*\\.png?") ) ) continue;
        
		int size = file_path.size();
		if(file_path.substr(size-3) != "png") continue;

        l.push_back(file_path);
    }
    closedir(dp);
}

void ShadowGesture::vectorize(){
	FileStorage fs("../Data/train_sequense.yml", FileStorage::READ);

	FileNode seqs_fs = fs["seqs"];
	FileNodeIterator it = seqs_fs.begin(), it_end = seqs_fs.end();

	Mat mat_all;
	for( ; it != it_end; ++it)
	{
		vector<string> seq;
		(*it) >> seq;
		for( int i = 0; i < (int)seq.size(); i++ ){
			Mat image = imread(seq[i]);
			cvtColor(image, image, CV_BGR2GRAY);
			threshold( image, image, 100, 1,THRESH_BINARY );
			resize(image, image, Size(20,20));
			image = image.reshape(1,1);
			mat_all.push_back(image);
		}
	}
	fs.release();

	Mat mean;
	pca = PCA(mat_all, mean, CV_PCA_DATA_AS_ROW, 43);
	//Mat evals = pca.eigenvalues;
	//Mat evecs = pca.eigenvectors;
	Mat mat_pca = pca.project(mat_all);

	label_amount = 4;
	Mat labels;
	kmeans(mat_pca, label_amount, labels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0.1), 100, KMEANS_RANDOM_CENTERS, centers);

	FileStorage fc("../Data/centers.yml", FileStorage::WRITE);
	fc << "C" << centers;
	fc.release();

	int rows = labels.rows/label_amount;
	Mat label_vecs = labels.reshape(0, rows);

	FileStorage fl("../Data/label_vecs.yml", FileStorage::WRITE);
	fl << "L" << label_vecs;
	fl.release();
}

void ShadowGesture::trainHMM(){
	Mat label_vecs;
	FileStorage fl("../Data/label_vecs.yml", FileStorage::READ);
	fl["L"] >> label_vecs;
	fl.release();

    double TRGUESSdata[] = {0.5 , 0.5 , 0.0, 0.0 ,
                            0.0 , 0.5 , 0.5, 0.0 ,
                            0.0 , 0.0 , 0.5 , 0.5,
                            0.0 , 0.0 , 0.0 , 1.0};
    cv::Mat TRGUESS = cv::Mat(4,4,CV_64F,TRGUESSdata).clone();
    double EMITGUESSdata[] = {1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
							  1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 , 1.0/4.0 };
    cv::Mat EMITGUESS = cv::Mat(4,7,CV_64F,EMITGUESSdata).clone();
    double INITGUESSdata[] = {0.6 , 0.5 , 0.4, 0.1};
    cv::Mat INITGUESS = cv::Mat(1,4,CV_64F,INITGUESSdata).clone();

    CvHMM hmm;
    hmm.train(label_vecs,100,TRGUESS,EMITGUESS,INITGUESS);
    hmm.printModel(TRGUESS,EMITGUESS,INITGUESS);

	FileStorage fh("../Data/hmm_params.yml", FileStorage::WRITE);
	fh << "T" << TRGUESS;
	fh << "E" << EMITGUESS;
	fh << "I" << INITGUESS;
	fh.release();

	cout << label_vecs.rows << endl;
	cout << label_vecs.cols << endl;
	cout << label_vecs << endl;

    cv::Mat pstates,forward,backward;
    double logpseq;
    std::cout << "\n";
    for (int i=0;i<label_vecs.rows;i++)
    {
        //std::cout << "row " << i << ": " << label_vecs.row(i) << "\n";
        hmm.decode(label_vecs.row(i),TRGUESS,EMITGUESS,INITGUESS,logpseq,pstates,forward,backward);
        std::cout << "logpseq" << i << " " << logpseq << "\n";
    }
    std::cout << "\n";
}

void ShadowGesture::testHMM(string path){
	FileStorage fs(path, FileStorage::READ);

	FileNode seqs_fs = fs["seqs"];
	FileNodeIterator it = seqs_fs.begin(), it_end = seqs_fs.end();

	Mat seqs;
	for( ; it != it_end; ++it)
	{
		vector<string> seq;
		(*it) >> seq;
		for( int i = 0; i < (int)seq.size(); i++ ){
			Mat image = imread(seq[i]);
			cvtColor(image, image, CV_BGR2GRAY);
			threshold( image, image, 100, 1,THRESH_BINARY );
			resize(image, image, Size(20,20)); 
			image = image.reshape(1,1);
			seqs.push_back(image);
		}
	}
	fs.release();

	Mat label_vecs = getPointClusters(seqs);

	Mat TRANS;
	Mat EMIS;
	Mat INIT;
	FileStorage fh("../Data/hmm_params.yml", FileStorage::READ);
	fh["T"] >> TRANS;
	fh["E"] >> EMIS;
	fh["I"] >> INIT;
	fh.release();

    CvHMM hmm;
    Mat pstates,forward,backward;
    double logpseq;
    cout << "\n";
    for (int i=0;i<label_vecs.rows;i++)
    {
        //cout << "row " << i << ": " << labels.row(i) << "\n";
        hmm.decode(label_vecs.row(i),TRANS,EMIS,INIT,logpseq,pstates,forward,backward);
        cout << "logpseq" << i << " " << logpseq << "\n";
    }
    cout << "\n";


    cv::Mat estates;
    std::cout << "\n";
    for (int i=0;i<label_vecs.rows;i++)
    {
        std::cout << i << ": ";
        hmm.viterbi(label_vecs.row(i),TRANS,EMIS,INIT,estates);
        for (int i=0;i<estates.cols;i++)
            std::cout << estates.at<int>(0,i);
        std::cout << "\n";
    }
    std::cout << "\n";


	cout << "Done!\n";
	for(;;){
		int key = waitKey(30);
		if(key == 27)//ESC
			break;
	}
}

Mat ShadowGesture::getPointClusters(Mat& seqs){
	FileStorage fc("../Data/centers.yml", FileStorage::READ);
	fc["C"] >> centers;
	fc.release();

	Mat mat_pca = pca.project(seqs);

	const int const clust_num = centers.rows;
	Mat labels;

	for(int r=0; r<mat_pca.rows; ++r ){
		vector<double> dist_from_clust;
		for(int k=0; k<clust_num; ++k ){
			Mat diff = centers.row(k)-mat_pca.row(r);
			dist_from_clust.push_back(sqrt(sum(diff*diff.t())[0]));
		}
		int label = min_element(dist_from_clust.begin(),dist_from_clust.end()) - dist_from_clust.begin();
		labels.push_back(label);
	}

	int rows = labels.rows/clust_num;
	Mat label_vecs = labels.reshape(0, rows);

	cout << label_vecs.rows << endl;
	cout << label_vecs.cols << endl;
	cout << label_vecs << endl;

	//for(int r=0; r<mat_pca.rows; ++r ){
	//	for(int c=0; c<mat_pca.cols; ++c ){
	//		vector<float> dist_from_clust;
	//		for(int k=0; k<clust_num; ++k ){
	//		dist_from_clust.push_back(0);
	//			float sqr_sum = 0;
	//			for(int f=0; f<centers.cols; ++f ){
	//				float cdist = (centers.at<float>(k,f) - mat_pca.at<float>(r,c));
	//				sqr_sum += cdist*cdist;
	//			}
	//			dist_from_clust[k] = sqrt(sqr_sum);
	//		}
	//		float label = min_element(dist_from_clust.begin(),dist_from_clust.end()) - dist_from_clust.begin();
	//		labels.push_back(label);
	//	}
	//}

	return label_vecs;
}

void ShadowGesture::FindConvexityDefects(){

	vector

	for(int i=0; i<paths.size(); i++){
	Mat img = imread("../Data/images/hand127_0.png");
	imshow("image", img);


	cvtColor(img, img, CV_RGB2GRAY);
	threshold(img, img, 100, 255, THRESH_BINARY);

	Mat img_c = img.clone();

	// find touch points
	vector< vector<Point> > contours;
	findContours(img_c, contours, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_TC89_L1);

	vector<vector<int> >hull( contours.size() );
	vector<vector<Point> >hull_points( contours.size() );
	vector<Vec4i>defects( contours.size() );
	for( int i = 0; i < contours.size(); i++ ){
		convexHull( contours[i], hull[i]);
		convexHull( Mat(contours[i]), hull_points[i]);
		convexityDefects(contours[i], hull[i], defects);
	}

	/// Draw contours + hull results
	RNG rng(12345);
	Mat drawing = Mat::zeros( img.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		drawContours( drawing, contours, i, Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull_points, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );

		vector<Point>& points = contours[i];
		cout << "number of point in the contour:" << points.size() << endl;
		for(int j=0; j<defects.size(); j++){
			Vec4i& defect = defects[j];
			int start_index = defects[j][0];
			int end_index = defects[j][1];
			int farthest_pt_index = defects[j][2];
			int fixpt_depth = defects[j][3];
			cout << "start_index: " << start_index << endl;
			cout << "end_index: " << end_index << endl;
			cout << "farthest_pt_index: " << farthest_pt_index << endl;
			cout << "fixpt_depth: " << fixpt_depth << endl;
			cout << "--------------------------------------" << endl;
			if(fixpt_depth>1000){
				Point middle((points[start_index].x+points[end_index].x)/2,
						(points[start_index].y+points[end_index].y)/2);
				line(drawing, points[start_index], points[end_index], Scalar(255,0,255), 2, 8);
				line(drawing, middle, points[farthest_pt_index], Scalar(0,255,255), 2, 8);
				circle(drawing, points[farthest_pt_index], 3, Scalar(0,255,255), -1, 8);
			}
		}
	}

	/// Show in a window
	namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
	imshow( "Hull demo", drawing );


	//Mat hull;
	//Mat conv;
	//convexHull(contours[0], hull);
	//cout << hull << endl;
	//convexityDefects(contours[0], hull, conv);
	//imshow("convexityDefects", conv);

		//for (unsigned int i=0; i<contours.size(); i++) {
		//	Mat contourMat(contours[i]);
		//	Rect brect = boundingRect(contourMat);
		//	if ( contourArea(contourMat) > 3000 ) {
		//		Rect urect(brect.x, brect.y, brect.width, roi_height);
		//		rectangle(screen_rgb, urect, Scalar(0,255,255));
		//		if(urect.y<screen_height-roi_height){
		//			vector< vector<Point2i> > icontours;
		//			findContours(screen_cc(urect), icontours, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE, brect.tl());
		//			for (unsigned int i=0; i<icontours.size(); i++) {
		//				Mat iContourMat(icontours[i]);
		//				Rect hrect = boundingRect(iContourMat);
		//				Rect frect;
		//				if(hrect.width<roi_height){
		//					Point p(hrect.x+hrect.width/2-roi_height/2, hrect.y);
		//					frect = Rect(p.x, p.y, roi_height, roi_height);
		//				} else if(hrect.x == urect.x+1){
		//					frect = Rect(hrect.x, hrect.y, roi_height, roi_height);
		//				} else {
		//					//stringstream s;
		//					//s << hrect.x-urect.x;
		//					//putText(screen_rgb, s.str(), Point(100,100), 1, 1, Scalar(0,0,255));
		//					frect = Rect(urect.x+urect.width-roi_height, urect.y, roi_height, roi_height);
		//				}
		//				rectangle(screen_rgb, frect, Scalar(0,0,255));
		//				if(frect.y<screen_height-roi_height && frect.x<screen_width-roi_height && frect.y>0 && frect.x>0){
		//					handRects.push_back(screen_hand(frect));
		//				}
		//			}
		//		}
		//	}
		//}










	//Mat bin;
	//threshold(img, bin, 100, 255, CV_THRESH_BINARY);
	//imshow("threshold", bin);

	//Mat bin8;
	//bin.convertTo(bin8, CV_32SC1);
	//imshow("bin8", bin8);

	//vector< vector<Point> > contours;
	//Mat hull;
	//Mat conv;
	//findContours(bin8, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//findContours(bin8, contours, CV_RETR_FLOODFILL, CV_CHAIN_APPROX_SIMPLE);
	//convexHull(contours, hull);
	//convexityDefects(contours[0], hull, conv);
	//imshow("convexityDefects", conv);

	waitKey(0);

}