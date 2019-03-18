#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
using namespace std;
using namespace cv;
#define PI 3.14159265 

vector<float>parametric_fit(vector<Point2f> sample_points, int n, Mat frame, bool draw);
vector<Point2f> find_inliers(vector<Point2f> sample_points, float m, float b, float d);
vector<Point2f> ransac_linesegment(vector<Point2f> edgepoints, int n, Mat frame, Mat original,int inlier_number);
vector<Point2f> remove_line_outliers(vector<Point2f> global_sampled);

struct myclass {
bool operator() (Point2f i, Point2f j) { return (pow(i.x,2) + pow(i.y,2) < pow(j.x,2) + pow(j.y,2));}
} myobject;

//define least square fitting function
vector<float>parametric_fit(vector<Point2f> sample_points, int n, Mat frame, bool draw){

//from the sample points, get the parametric equation of the line
	vector<float> parametric;
	float mean_x = 0, mean_y = 0, sum_x = 0, sum_y=0, upper=0, lower=0,lower_n, m= 0, m_y = 0, b =0, theta=0, A = 0, B=0;
	for(size_t i = 0; i<n;i++){
		sum_x += sample_points[i].x;sum_y += sample_points[i].y;
		mean_x = sum_x/n; mean_y = sum_y/n;
	}

	//get m and b
	for(size_t i = 0; i<n;i++){
		upper +=(sample_points[i].x - mean_x)*(sample_points[i].y - mean_y);
		lower +=pow((sample_points[i].x - mean_x),2);
		lower_n +=pow((sample_points[i].y - mean_y),2);
	}
	m = upper/lower; b = mean_y - m*mean_x;theta = atan(m);
	parametric.push_back(m);parametric.push_back(b);

	//drawing line
	if(draw){
		Point reference, point_1, point_2;
		reference.x = roundf(mean_x);reference.y = roundf(mean_y);
		vector<float> distances;
		for(size_t i = 0; i<sample_points.size(); i++){
			distances.push_back(sqrt(pow(sample_points[i].x - reference.x,2) + pow(sample_points[i].y - reference.y,2)));
		}
		float length = *max_element(distances.begin(), distances.end());
		point_1.x = reference.x + roundf(length * cos(theta)); point_1.y = reference.y + roundf(length * sin(theta));
		point_2.x = reference.x - roundf(length * cos(theta)); point_2.y = reference.y - roundf(length * sin(theta));
		line(frame,reference, point_2, (0,0,255),2);line(frame,reference, point_1, (0,0,255),2);
	}
	return parametric;
}

//inlier checking and finding function
vector<Point2f> find_inliers(vector<Point2f> unsampled, float m, float b, float d){
	
//for non-sampled points, calculate the distances to unsampled points
	vector<Point2f> inliers;
	for(size_t i = 0 ; i<unsampled.size(); i++){
		float dist = (abs(m*unsampled[i].x - unsampled[i].y + b))/(sqrt(pow(m,2)+1));
		if (dist < d){ 	//distance threshold
			inliers.push_back(unsampled[i]);
		}
	}
	return inliers;
}

//line detection function
vector<Point2f> ransac_linesegment(vector<Point2f> edgepoints, int n, Mat frame, Mat original, int inlier_number){
	vector< Point2f > global_sampled, global_unsampled, temp_inliers;
	Mat canny_output; int low_threshold = 350; int ratio = 3;
	int iterations = 0;
	while(1){
		Canny(frame, canny_output, low_threshold, low_threshold*ratio, 5);
		//draw samples and separate sampled from unsampled points
		vector<Point2f> sampled; vector<Point2f> unsampled = edgepoints;
		for(size_t i=0;i<n;i++){//n sample points  matrix size is 640 x 480
			Point point;
			int selector = rand() % edgepoints.size();
			point.x = edgepoints[selector].x;
			point.y = edgepoints[selector].y;
			sampled.push_back(point);
			unsampled.erase(unsampled.begin() + selector);
		}

		// fit the line using the sampled points
		vector<float> parametric = parametric_fit(sampled, sampled.size(), canny_output, 1);
		// check distance the unsampled points and generate inliers
		vector<Point2f> inliers = find_inliers(unsampled, parametric[0], parametric[1], 1);
		// when the inliers are above inlier_number, copy them to global_sampled points  and exit loop
		if(inliers.size() > inlier_number){
			global_sampled = sampled; global_unsampled = unsampled;temp_inliers = inliers;
			namedWindow("first_guess", WINDOW_NORMAL);imshow("first_guess",canny_output);waitKey(1);
			break;
		}
		iterations += 1;
	}

	//update the sample points and redraw the line aka second loop
	if (global_sampled.size() > 1 ){
		int min_inlier_size = 100;
		while(min_inlier_size > 200){
			for(vector<Point2f>::iterator iter = temp_inliers.begin() ; iter < temp_inliers.end();iter ++){
				global_sampled.push_back(*iter);
				global_unsampled.erase(remove(global_unsampled.begin(), global_unsampled.end(), *iter), global_unsampled.end());
			}
			vector<float> new_parametric = parametric_fit(global_sampled, global_sampled.size(), original, 0);
			vector<Point2f> new_inliers = find_inliers(global_unsampled, new_parametric[0], new_parametric[1], 1);
			min_inlier_size = new_inliers.size(); std::cout<<min_inlier_size <<" \n";
			temp_inliers = new_inliers;
		}
		global_sampled = remove_line_outliers(global_sampled);
		vector<float> final_parametric = parametric_fit(global_sampled, global_sampled.size(), original, 1);
		// set the remaining unsampled points
		vector<Point2f> remaining = edgepoints;
		for(vector<Point2f>::iterator iter = global_sampled.begin(); iter < global_sampled.end(); iter ++ ){
			remaining.erase(remove(remaining.begin(), remaining.end(), *iter), remaining.end());
		}
	namedWindow("adapted fit", WINDOW_NORMAL);
	imshow("adapted fit",original);
	return remaining;
	}
	
//else{return global_unsampled;}	
}

//function for removing outliers along the fitted line
vector<Point2f> remove_line_outliers(vector<Point2f> global_sampled){
	
//get the mean
	Point2f mean; mean.x =0; mean.y=0;
	for(size_t i = 0; i < global_sampled.size(); i++){
		mean.x += global_sampled[i].x;mean.y += global_sampled[i].y;
	}
	mean.x = mean.x/global_sampled.size(); mean.y = mean.y/global_sampled.size();

	//get the standard deviation
	Point2f stdev; stdev.x = 0; stdev.y = 0;
	for(size_t i = 0; i < global_sampled.size(); i++){
		stdev.x += pow(global_sampled[i].x - mean.x,2); stdev.y += pow(global_sampled[i].y - mean.y,2);
	}
	stdev.x = sqrt(stdev.x/global_sampled.size()); stdev.y = sqrt(stdev.y/global_sampled.size());

	//choose x or y
	float dev; dev = stdev.x; if(stdev.y > stdev.x){dev = stdev.y;}
	for(size_t i = 0; i < global_sampled.size(); i++){
		if(global_sampled[i].x > mean.x + 2*dev){
			//global_sampled[i].x = roundf(mean.x); global_sampled[i].y = roundf(mean.y);
			global_sampled.erase(global_sampled.begin()+i); i = i-1;
		}
		if(global_sampled[i].x < mean.x - 2*dev){ 
			//global_sampled[i].x = roundf(mean.x); global_sampled[i].y = roundf(mean.y);
			global_sampled.erase(global_sampled.begin()+i); i = i-1;
		}
	}
	return global_sampled;
}

int main(int argc, char *argv[])
	{
	Mat frame,original, canny_output;
	srand(time(NULL));
	vector<Point2f> edgepoints;
	string filename = argv[1];
	frame = imread(filename);
	cout<<"image size "<<frame.size<<endl;
	original = frame.clone();
	blur(frame, frame, Size(5,5));
	int low_threshold = 100; int ratio = 1.5;
	Canny(frame,canny_output,low_threshold,low_threshold*ratio,5);

	///RANSAC
        int n = 2;//number of initial sampling points
	//get vector of non-zero intensity (edge) points
	for(size_t i = 0 ; i < canny_output.size().height;i++){
		for(size_t j = 0 ; j < canny_output.size().width;j++){
			int intensity = canny_output.at<uchar>(i,j);
			if(intensity != 0){
				Point point;
				point.x = j; point.y = i;
				edgepoints.push_back(point);
			}
		}
	}
	vector<Point2f> segment = edgepoints ; int counter = 0;
	while(1){
		segment = ransac_linesegment(segment, n, frame, original,30);
		counter += 1;
		if(segment.size() < 30){
			break;
		}
	}

	cout<<"edges detected  "<< counter<<endl;
	//setMouseCallback("adapted fit", CallBackFunc, NULL);
	waitKey(0);
}


