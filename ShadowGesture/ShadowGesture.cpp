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

void ShadowGesture::capture(string path){
	cout << "Staring ShadowGesture..." << endl;

	cap.open(path);
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

		//for (int i=0 ; i<handRects.size() ; ++i) {
		//	stringstream s;
		//	s << i;
		//	stringstream f;
		//	f << cap.get(CV_CAP_PROP_POS_FRAMES);
		//	imshow("hand"+s.str(), handRects[i]);
		//	imwrite("../Data/images/hand"+f.str()+"_"+s.str()+".png", handRects[i]);
		//}


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

		//imshow("image", image);

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

void ShadowGesture::vectorize(string path){
	FileStorage fs(path, FileStorage::READ);

	FileNode seqs_fs = fs["seqs"];
	FileNodeIterator it = seqs_fs.begin(), it_end = seqs_fs.end();

	Mat mat_all;
	for( ; it != it_end; ++it)
	{
		vector<string> seq;
		(*it) >> seq;
		for( int i = 0; i < (int)seq.size(); i++ ){
			mat_all.push_back(FindConvexityDefects(seq[i]));
		}
	}
	fs.release();

	int num_frames = 4;
	int num_observations = mat_all.rows;
	int num_features = mat_all.cols;

	int sz[] = {num_frames, num_observations/num_frames, num_features};
	Mat mat_all_3d(3, sz, CV_32F);
	for(int n=0; n<num_frames; n++){
		for(int i=0; i<num_observations/num_frames; i++){
			for(int j=0; j<num_features; j++){
				mat_all_3d.at<float>(n,i,j) = mat_all.at<float>(num_frames*i + n, j);
			}
		}
	}

	Mat mean(num_frames, num_features, CV_32F);

	for(int n=0; n<num_frames; n++){
		for(int j=0; j<num_features; j++){
			mean.at<float>(n,j) = 0;
		}
	}

	for(int n=0; n<num_frames; n++){
		for(int i=0; i<num_observations/num_frames; i++){
			for(int j=0; j<num_features; j++){
				mean.at<float>(n,j) += mat_all_3d.at<float>(n,i,j);
			}
		}
	    mean.row(n) /= num_observations;
	}

	num_hidden_states = 3;
	num_output_symbols = 4;
	
	Mat center_labels;
	kmeans(mean, num_hidden_states, center_labels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0.1), 100, KMEANS_RANDOM_CENTERS, centers);

	FileStorage fc("../Data/centers.yml", FileStorage::WRITE);
	fc << "C" << centers;
	fc.release();

	Mat label_vecs = getPointClusters(mat_all_3d, num_frames, num_observations/num_frames, num_features);

	FileStorage fl("../Data/label_vecs.yml", FileStorage::WRITE);
	fl << "L" << label_vecs;
	fl.release();
}

Mat ShadowGesture::getPointClusters(Mat& seqs, int num_frames, int num_observations, int num_features){
	Mat centroids;
	FileStorage fc("../Data/centers.yml", FileStorage::READ);
	fc["C"] >> centroids;
	fc.release();

	Mat labels(num_observations,num_frames, CV_32S);
	int K = centers.rows;

	for(int r=0; r<num_frames; r++){
		for(int c=0; c<num_observations; c++){
	        vector<int> temp;
			for(int j=0; j<K; j++){
				float sq = 0;
				for(int d=0; d<num_features; d++){
	                sq += (centroids.at<float>(j,1) - seqs.at<float>(r,c,d))*(centroids.at<float>(j,1) - seqs.at<float>(r,c,d));
				}
				temp.push_back(sqrt(sq));
			}
			labels.at<int>(c,r) = min_element(temp.begin(),temp.end()) - temp.begin();
		}
	}

	int sz[] = {num_observations,num_frames};
	labels = labels.reshape(1,2,sz);

	return labels;
}

void ShadowGesture::trainHMM(string path){
	Mat label_vecs;
	FileStorage fl(path, FileStorage::READ);
	fl["L"] >> label_vecs;
	fl.release();

    double TRGUESSdata[] = {2.0/3.0 , 1.0/3.0 , 0.0/3.0  , 0.0/3.0,
                            0.0/3.0 , 2.0/3.0 , 1.0/3.0  , 0.0/3.0,
                            0.0/3.0 , 0.0/3.0 , 2.0/3.0  , 1.0/3.0,
                            0.0/3.0 , 0.0/3.0 , 0.0/3.0  , 3.0/3.0};
    cv::Mat TRGUESS = cv::Mat(4,4,CV_64F,TRGUESSdata).clone();
    double EMITGUESSdata[] = {1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 ,
                              1.0/4.0 , 1.0/4.0 , 1.0/4.0 };
    cv::Mat EMITGUESS = cv::Mat(4,3,CV_64F,EMITGUESSdata).clone();
    double INITGUESSdata[] = {0.6  , 0.2 , 0.2, 0.2};
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
	cout << label_vecs.channels() << endl;
	cout << label_vecs.type() << endl;

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

	double sumLik = 0;
	double minLik = numeric_limits<double>::infinity();
	for (int j=0;j<label_vecs.rows;j++)
	{
		double lik = prHmm(label_vecs.row(j));
		if (lik < minLik){
			minLik = lik;
		}
		sumLik = sumLik + lik;
	}
	double gestureRecThreshold = 2.0*sumLik/label_vecs.rows;

	cout << "gestureRecThreshold: " << gestureRecThreshold << endl;
}

double ShadowGesture::prHmm(Mat& o){
	Mat TRANS;
	Mat EMIS;
	Mat INIT;
	FileStorage fh("../Data/hmm_params.yml", FileStorage::READ);
	fh["T"] >> TRANS;
	fh["E"] >> EMIS;
	fh["I"] >> INIT;
	fh.release();

	int n = TRANS.cols;
	int T = o.cols;

	Mat m(T,n,CV_64F);
    for (int i=0;i<n;i++){
	    m.at<double>(0,i)=EMIS.at<double>(i,o.at<int>(0,0))*INIT.at<double>(0,i);
	}

    for (int t=0;t<T-1;t++){
	    for (int j=0;j<n;j++){
	        double z = 0;
		    for (int i=0;i<n;i++){
	            z=z+TRANS.at<double>(i,j)*m.at<double>(t,i);
			}
	        m.at<double>(t+1,j)=z*EMIS.at<double>(j,o.at<int>(t+1));
		}
	}

	double p = 0;
    for (int i=0;i<n;i++){
	    p=p+m.at<double>(T-1,i);
	}

	return log(p);
}

void ShadowGesture::testHMM(string path){
	FileStorage fs(path, FileStorage::READ);

	FileNode seqs_fs = fs["seqs"];
	FileNodeIterator it = seqs_fs.begin(), it_end = seqs_fs.end();

	Mat mat_all;
	for( ; it != it_end; ++it)
	{
		vector<string> seq;
		(*it) >> seq;
		for( int i = 0; i < (int)seq.size(); i++ ){
			mat_all.push_back(FindConvexityDefects(seq[i]));
		}
	}
	fs.release();

	int num_frames = 4;
	int num_observations = mat_all.rows;
	int num_features = mat_all.cols;

	int sz[] = {num_frames, num_observations/num_frames, num_features};
	Mat mat_all_3d(3, sz, CV_32F);
	for(int n=0; n<num_frames; n++){
		for(int i=0; i<num_observations/num_frames; i++){
			for(int j=0; j<num_features; j++){
				mat_all_3d.at<float>(n,i,j) = mat_all.at<float>(num_frames*i + n, j);
			}
		}
	}

	Mat label_vecs = getPointClusters(mat_all_3d, num_frames, num_observations/num_frames, num_features);

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
    //cout << "\n";
    for (int i=0;i<label_vecs.rows;i++)
    {
        //cout << "row " << i << ": " << labels.row(i) << "\n";
        hmm.decode(label_vecs.row(i),TRANS,EMIS,INIT,logpseq,pstates,forward,backward);
        cout << "logpseq" << i << " " << logpseq << "\n";
    }
    cout << "\n";

//	waitKeyPress();

    //cv::Mat estates;
    ////std::cout << "\n";
    //for (int i=0;i<label_vecs.rows;i++)
    //{
    //    //std::cout << i << ": ";
    //    hmm.viterbi(label_vecs.row(i),TRANS,EMIS,INIT,estates);
    //    //for (int i=0;i<estates.cols;i++)
    //        //std::cout << estates.at<int>(0,i);
    //    //std::cout << "\n";
    //}
    ////std::cout << "\n";

}

Mat ShadowGesture::FindConvexityDefects(string path){
	Mat img = imread(path);
	//imshow("image", img);

	cvtColor(img, img, CV_RGB2GRAY);
	threshold(img, img, 100, 255, THRESH_BINARY);
//	resize(img, img, Size(20,20));

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

	Mat features(5,1,CV_32F);
	features.zeros(5,1,CV_32F);

	/// Draw contours + hull results
	RNG rng(12345);
	Mat drawing = Mat::zeros( img.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		drawContours( drawing, contours, i, Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull_points, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );

		vector<Point>& points = contours[i];
		//cout << "number of point in the contour:" << points.size() << endl;
		int largest_defect = -1;
		int largest_fixpt_depth = -1;
		float s3_ang = -1;
		for(int j=0; j<defects.size(); j++){
			Vec4i& defect = defects[j];
			int start_index = defect[0];
			int end_index = defect[1];
			int farthest_pt_index = defect[2];
			int fixpt_depth = defect[3];
			//cout << "start_index: " << start_index << endl;
			//cout << "end_index: " << end_index << endl;
			//cout << "farthest_pt_index: " << farthest_pt_index << endl;
			//cout << "fixpt_depth: " << fixpt_depth << endl;
			//cout << "--------------------------------------" << endl;
			if(largest_fixpt_depth<fixpt_depth){
				largest_fixpt_depth = fixpt_depth;
				largest_defect = j;

				Point2f s3 = points[start_index]-points[end_index];
				mag_ang ma_s3 = calcMagAng(s3);
				s3_ang = ma_s3.angle;
			}
		}
		if(largest_defect>=0 && s3_ang>230){
			//cout << s3_ang << endl;
			Vec4i& defect = defects[largest_defect];
			int start_index = defect[0];
			int end_index = defect[1];
			int farthest_pt_index = defect[2];
			int fixpt_depth = defect[3];
			Point mid_point((points[start_index].x+points[end_index].x)/2,
						 (points[start_index].y+points[end_index].y)/2);
			
			line(drawing, points[start_index], points[end_index], Scalar(255,0,255), 2, 8);
			line(drawing, points[start_index], points[farthest_pt_index], Scalar(255,0,255), 2, 8);
			line(drawing, points[end_index], points[farthest_pt_index], Scalar(255,0,255), 2, 8);
			line(drawing, mid_point, points[farthest_pt_index], Scalar(0,255,255), 2, 8);
			circle(drawing, points[farthest_pt_index], 3, Scalar(0,255,255), -1, 8);

			Point2f s1 = points[start_index]-points[farthest_pt_index];
			Point2f s2 = points[end_index]-points[farthest_pt_index];
			Point2f mid = mid_point - points[farthest_pt_index];

			mag_ang ma_s1 = calcMagAng(s1);
			mag_ang ma_s2 = calcMagAng(s2);
			mag_ang ma_mid = calcMagAng(mid);
			//cout << "ang: " << ma_s1.angle-ma_s2.angle << endl;
			//cout << "ratio: " << ma_s1.magnitude/ma_s2.magnitude << endl;
			//cout << "mag: " << ma_mid.angle << endl;

			features.at<float>(0) = ma_s1.angle-ma_s2.angle;
			features.at<float>(1) = ma_s1.magnitude/ma_s2.magnitude;
			features.at<float>(2) = ma_mid.angle;
			features.at<float>(3) = s3_ang - ma_s1.angle;
			features.at<float>(4) = ma_mid.magnitude;
		}
	}

	///// Show in a window
	//namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
	//imshow( "Hull demo", drawing );

//	waitKey(0);
	
	return features.t();

}

ShadowGesture::mag_ang ShadowGesture::calcMagAng(Point2f& p){
	Mat mag, ang;
	cartToPolar(p.x, p.y, mag, ang, true);
	mag_ang ma = {mag.at<double>(0), ang.at<double>(0)};
	return ma;
}


void ShadowGesture::recognizeGesture(string path){
	cout << "Staring recognizeGesture..." << endl;

	cap.open(path);
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

	roi_height = 100;
	screen = Mat(screen_height, screen_width, CV_8UC1);

	screen_mat = findHomography(video_corners, screen_corners);

	Mat mat_all;

	Mat TRANS;
	Mat EMIS;
	Mat INIT;
	FileStorage fh("../Data/hmm_params.yml", FileStorage::READ);
	fh["T"] >> TRANS;
	fh["E"] >> EMIS;
	fh["I"] >> INIT;
	fh.release();

    CvHMM hmm;

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

		if(play_all){
			for (int i=0 ; i<handRects.size() ; ++i) {
				mat_all.push_back(FindConvexityDefects(handRects[i]));
				if(mat_all.rows>4){
					mat_all = mat_all(Range(1, 5),Range::all());

					int num_frames = 4;
					int num_observations = mat_all.rows;
					int num_features = mat_all.cols;

					int sz[] = {num_frames, num_observations/num_frames, num_features};
					Mat mat_all_3d(3, sz, CV_32F);
					for(int n=0; n<num_frames; n++){
						for(int i=0; i<num_observations/num_frames; i++){
							for(int j=0; j<num_features; j++){
								mat_all_3d.at<float>(n,i,j) = mat_all.at<float>(num_frames*i + n, j);
							}
						}
					}

					Mat label_vecs = getPointClusters(mat_all_3d, num_frames, num_observations/num_frames, num_features);

					Mat pstates,forward,backward;
					double logpseq;
					//cout << "\n";
					for (int i=0;i<label_vecs.rows;i++)
					{
						//cout << "row " << i << ": " << labels.row(i) << "\n";
						hmm.decode(label_vecs.row(i),TRANS,EMIS,INIT,logpseq,pstates,forward,backward);
						cout << "logpseq" << i << " " << logpseq << "\n";
					}
					cout << "-------------\n";
				}
				play_all = false;
			}
		}

		imshow("screen", screen_rgb);

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


Mat ShadowGesture::FindConvexityDefects(Mat& img){
	Mat img_c = img.clone();

	// find touch points
	vector< vector<Point> > contours;
	findContours(img_c, contours, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_TC89_L1);

	vector<vector<int> >hull( contours.size() );
	vector<vector<Point> >hull_points( contours.size() );
	vector< vector<Vec4i> > defects( contours.size() );
	for( int i = 0; i < contours.size(); i++ ){
		convexHull( contours[i], hull[i]);
		convexHull( Mat(contours[i]), hull_points[i]);
		if(contours[i].size()>2)
			convexityDefects(contours[i], hull[i], defects[i]);
	}

	Mat features(5,1,CV_32F);
	features.zeros(5,1,CV_32F);

	/// Draw contours + hull results
	RNG rng(12345);
	Mat drawing = Mat::zeros( img.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		drawContours( drawing, contours, i, Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull_points, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );

		vector<Point>& points = contours[i];
		int largest_defect = -1;
		int largest_fixpt_depth = -1;
		float s3_ang = -1;
		for(int j=0; j<defects[i].size(); j++){
			Vec4i& defect = defects[i][j];
			int start_index = defect[0];
			int end_index = defect[1];
			int farthest_pt_index = defect[2];
			int fixpt_depth = defect[3];
			//cout << "start_index: " << start_index << endl;
			//cout << "end_index: " << end_index << endl;
			//cout << "farthest_pt_index: " << farthest_pt_index << endl;
			//cout << "fixpt_depth: " << fixpt_depth << endl;
			//cout << "--------------------------------------" << endl;
			if(largest_fixpt_depth<fixpt_depth){
				largest_fixpt_depth = fixpt_depth;
				largest_defect = j;

				Point2f s3 = points[start_index]-points[end_index];
				mag_ang ma_s3 = calcMagAng(s3);
				s3_ang = ma_s3.angle;
			}
		}
		if(largest_defect>=0 && s3_ang>230){
			//cout << s3_ang << endl;
			Vec4i& defect = defects[i][largest_defect];
			int start_index = defect[0];
			int end_index = defect[1];
			int farthest_pt_index = defect[2];
			int fixpt_depth = defect[3];
			Point mid_point((points[start_index].x+points[end_index].x)/2,
						 (points[start_index].y+points[end_index].y)/2);
			
			line(drawing, points[start_index], points[end_index], Scalar(255,0,255), 2, 8);
			line(drawing, points[start_index], points[farthest_pt_index], Scalar(255,0,255), 2, 8);
			line(drawing, points[end_index], points[farthest_pt_index], Scalar(255,0,255), 2, 8);
			line(drawing, mid_point, points[farthest_pt_index], Scalar(0,255,255), 2, 8);
			circle(drawing, points[farthest_pt_index], 3, Scalar(0,255,255), -1, 8);

			Point2f s1 = points[start_index]-points[farthest_pt_index];
			Point2f s2 = points[end_index]-points[farthest_pt_index];
			Point2f mid = mid_point - points[farthest_pt_index];

			mag_ang ma_s1 = calcMagAng(s1);
			mag_ang ma_s2 = calcMagAng(s2);
			mag_ang ma_mid = calcMagAng(mid);
			//cout << "ang: " << ma_s1.angle-ma_s2.angle << endl;
			//cout << "ratio: " << ma_s1.magnitude/ma_s2.magnitude << endl;
			//cout << "mag: " << ma_mid.angle << endl;

			features.at<float>(0) = ma_s1.angle-ma_s2.angle;
			features.at<float>(1) = ma_s1.magnitude/ma_s2.magnitude;
			features.at<float>(2) = ma_mid.angle;
			features.at<float>(3) = s3_ang - ma_s1.angle;
			features.at<float>(4) = ma_mid.magnitude;
		}
	}

	/// Show in a window
	namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
	imshow( "Hull demo", drawing );

//	waitKey(0);
	
	return features.t();

}


void ShadowGesture::convertDataToOctaveCVS(string path){
	FileStorage fs(path, FileStorage::READ);

	FileNode seqs_fs = fs["seqs"];
	FileNodeIterator it = seqs_fs.begin(), it_end = seqs_fs.end();

	Mat mat_all;
	for( ; it != it_end; ++it)
	{
		vector<string> seq;
		(*it) >> seq;
		for( int i = 0; i < (int)seq.size(); i++ ){
			mat_all.push_back(FindConvexityDefects(seq[i]));
		}
	}
	fs.release();

	int seq_of_frames = 4;
	for(int i=0 ; i<mat_all.cols ; i++){
		stringstream i_str;
		i_str << i;

		Mat feature = mat_all.col(i).clone();
		feature = feature.reshape(0,mat_all.rows/seq_of_frames).t();

		stringstream csv_text;
		csv_text << format(feature , "csv");

		string csv_path = "../Data/ForOctave/feature"+i_str.str()+".csv";
		FILE* file = fopen(csv_path.c_str(),"w");
		fputs(csv_text.str().c_str(), file);
		fclose(file);
	}

	waitKeyPress();
}

void ShadowGesture::convertBinaryDataToOctaveCVS(string train_path, string test_path){
	FileStorage fs(train_path, FileStorage::READ);

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

	mat_all = mat_all;

	cout << mat_all.rows << endl;
	cout << mat_all.cols << endl;

	Mat mean;
	pca = PCA(mat_all, mean, CV_PCA_DATA_AS_ROW, 43);
	cout << "pca.mean.rows: " << pca.mean.rows << endl;
	cout << "pca.mean.cols: " << pca.mean.cols << endl;
	Mat mat_pca = pca.project(mat_all);
	cout << mat_pca.rows << endl;
	cout << mat_pca.cols << endl;

	int seq_of_frames = 4;
	for(int i=0 ; i<mat_pca.cols ; i++){
		stringstream i_str;
		i_str << i;

		Mat feature = mat_pca.col(i).clone();
		feature = feature.reshape(0,mat_pca.rows/seq_of_frames).t();

		stringstream csv_text;
		csv_text << format(feature , "csv");

		string csv_path = "../Data/ForOctave/binary/train/feature"+i_str.str()+".csv";
		FILE* file = fopen(csv_path.c_str(),"w");
		fputs(csv_text.str().c_str(), file);
		fclose(file);
	}


	fs = FileStorage(test_path, FileStorage::READ);

	seqs_fs = fs["seqs"];
	it = seqs_fs.begin();
	it_end = seqs_fs.end();

	Mat mat_all_test;
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
			mat_all_test.push_back(image);
		}
	}
	fs.release();

	cout << mat_all_test.rows << endl;
	cout << mat_all_test.cols << endl;

	Mat mat_pca_test = pca.project(mat_all_test);
	cout << mat_pca_test.rows << endl;
	cout << mat_pca_test.cols << endl;

	for(int i=0 ; i<mat_pca_test.cols ; i++){
		stringstream i_str;
		i_str << i;

		Mat feature = mat_pca_test.col(i).clone();
		feature = feature.reshape(0,mat_pca_test.rows/seq_of_frames).t();

		stringstream csv_text;
		csv_text << format(feature , "csv");

		string csv_path = "../Data/ForOctave/binary/test/feature"+i_str.str()+".csv";
		FILE* file = fopen(csv_path.c_str(),"w");
		fputs(csv_text.str().c_str(), file);
		fclose(file);
	}

	waitKeyPress();
}

void ShadowGesture::waitKeyPress(){
	Mat klum(5,1,CV_32F);
	klum.zeros(5,1,CV_32F);
	imshow( "klum",  klum);

	waitKey(0);
}