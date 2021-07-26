#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <unistd.h>
#include <Eigen/Dense>

#define FRAMES_PATH "/home/marcus/Workspace/Coding/Python/bm_frames/"

using namespace cv;
using namespace std;

// camera
int FOCAL_LENGTH = 460;
const int ROW = 720;
const int COL = 1280;
// default cam info from 17/07/21
const double fx = 616.8896484375;
const double fy = 751.7689208984375;
const double cx = 656.7361546622124;
const double cy = 331.3465453109238;

static void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P)
{

    double m_inv_K11 = 1.0 / fx;
    double m_inv_K13 = cx / fx;
    double m_inv_K22 = 1.0 / fy;
    double m_inv_K23 = -cy / fy;
    double mx_d, my_d, mx_u, my_u;

    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;

    // no distortion
    mx_u = mx_d;
    my_u = my_d;

    P << mx_u, my_u, 1.0;
}

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

int main()
{

    std::string img_path(FRAMES_PATH);
    char img_id[12];
    int count = 0;
    sprintf(img_id, "%06d", count);
    cout << img_id << endl;

    cv::Mat cur_img, prev_img;
    vector<Point2f> cur_pts, prev_pts, new_pts;

    vector<Point2f> prev_un_pts, cur_un_pts;
    Mat mask;
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    int MAX_FEATURES = 100;
    Mat finalR, finalT;
    finalR = Mat::eye(3, 3, CV_64F);
    finalT = Mat::ones(3, 1, CV_64F);
    // finalT = Mat::zeros()
    Eigen::Vector3d prevT(0, 0, 0);
    while (1)
    {
        sprintf(img_id, "%06d", count);
        img_path = std::string(FRAMES_PATH) + std::string(img_id) + ".png";
        cout << "reading image: " << img_path << endl;

        if (count == 0)
        {
            prev_img = imread(img_path, IMREAD_GRAYSCALE);
            goodFeaturesToTrack(prev_img, prev_pts, 100, 0.3, 7, Mat(), 7, false, 0.04);
            mask = Mat::zeros(prev_img.size(), prev_img.type());
            count++;
            continue;
        }

        cur_img = imread(img_path, IMREAD_GRAYSCALE);

        // Process here
        vector<uchar> status;
        vector<float> err;

        calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, Size(11, 11), 3);

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        // reduceVector(ids, status);
        // reduceVector(cur_un_pts, status);
        // reduceVector(track_cnt, status);

        for (uint i = 0; i < cur_pts.size(); i++)
        {
            line(mask, prev_pts[i], cur_pts[i], colors[i], 2);
            circle(cur_img, cur_pts[i], 5, colors[i], -1);
        }

        // Project and match points
        vector<cv::Point2f> un_prev_pts(prev_pts.size()), un_cur_pts(cur_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + 1280 / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        // Mat transform = findFundamentalMat(un_prev_pts,un_cur_pts,FM_RANSAC,1.0,0.99,status);
        double camera_param[] = {fx, 0, cx, 0, fy, cy, 0, 0, 0, 1};

        const Mat CameraMatrix = Mat(3, 3, CV_64FC1, camera_param);

        Mat EMatrix = findEssentialMat(un_prev_pts, un_cur_pts, CameraMatrix, FM_RANSAC, 0.99, 1.0);
        Mat R, T;
        recoverPose(EMatrix, un_prev_pts, un_cur_pts, CameraMatrix, R, T);

        Eigen::Vector3d currT(T.at<double>(0), T.at<double>(1), T.at<double>(2));
        double scale_ = (currT - prevT).norm();

        if (scale_ > 0.1 && T.at<double>(0) > T.at<double>(1) && T.at<double>(0) > T.at<double>(2))
        {
            finalT = finalT + scale_ * R * T;
            finalR = finalR * R;
        }

        cout << "Rot: " << finalR << "\n T:" << finalT << endl;
        Mat img;
        add(cur_img, mask, img);
        imshow("Frame", img);
        if (waitKey(1) == 27)
            break; // stop capturing by pressing ESC

        // TODO heuristica para selecionar novos pontos ?
        std::cout << "total features: " << cur_pts.size();
        if (cur_pts.size() < MAX_FEATURES)
        {
            goodFeaturesToTrack(cur_img, new_pts, MAX_FEATURES - cur_pts.size(), 0.05, 40, Mat());
            std::cout << "total new features: " << new_pts.size();
            for (const auto &p : new_pts)
            {
                cur_pts.push_back(p);
            }
        }

        count++;
        prev_img = cur_img;
        prev_pts = cur_pts;
        mask = Mat::zeros(cur_img.size(), cur_img.type());
        usleep(100000);
    }
    // Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();

    // std::vector<KeyPoint> keypoints;
    // fast->detect(img,keypoints);

    // std::cout << "Detected: " << keypoints.size() << "keypoints\n";
    // Mat img2;

    // drawKeypoints(img,keypoints,img2);

    // imshow("window",img);
    // imshow("window2",img2);
    // int k = waitKey(0);
}