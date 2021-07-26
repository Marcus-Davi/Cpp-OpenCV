#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <unistd.h>
#include <Eigen/Dense>

#define FRAMES_PATH "/home/marcus/Workspace/Coding/Python/bm_frames/"
// #define FRAMES_PATH "/home/marcus/Workspace/Coding/Python/data_odometry_gray/dataset/sequences/00/image_0/"

using namespace cv;
using namespace std;

// camera
// int FOCAL_LENGTH = 460;
const int ROW = 720;
const int COL = 1280;
int MAX_FEATURES = 2000;
// default cam info from 17/07/21
const double fx = 616.8896484375;
const double fy = 751.7689208984375;
const double cx = 656.7361546622124;
const double cy = 331.3465453109238;

Vec3f rotationMatrixToEulerAngles(const Mat &R);

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
    vector<KeyPoint> keypts;

    // vector<Point2f> prev_un_pts, cur_un_pts;
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

    Mat finalR, finalT;
    finalR = Mat::eye(3, 3, CV_64F);
    finalT = Mat::ones(3, 1, CV_64F);
    // finalT = Mat::zeros()
    Eigen::Vector3d prevT(0, 0, 0);
    Eigen::Vector3d currT(0, 0, 0);

    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(25, true);

    while (1)
    {
        sprintf(img_id, "%06d", count);
        img_path = std::string(FRAMES_PATH) + std::string(img_id) + ".png";
        // cout << "reading image: " << img_path << endl;

        if (count == 0)
        {
            prev_img = imread(img_path, IMREAD_GRAYSCALE);
            fast->detect(prev_img, keypts);
            prev_pts.resize(keypts.size());
            int i = 0;
            for (const auto &pt : keypts)
            {
                prev_pts[i] = pt.pt;
                i++;
            }

            // goodFeaturesToTrack(prev_img, prev_pts, MAX_FEATURES, 0.05, 40, Mat());
            mask = Mat::zeros(prev_img.size(), prev_img.type());
            count++;
            continue;
        }

        cur_img = imread(img_path, IMREAD_GRAYSCALE);

        // Process here
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
        calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, Size(21, 21), 3, criteria);

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        // reduceVector(ids, status);
        // reduceVector(cur_un_pts, status);
        // reduceVector(track_cnt, status);

        // for (uint i = 0; i < cur_pts.size(); i++)
        // {
        //     line(mask, prev_pts[i], cur_pts[i], colors[i], 2);
        //     circle(cur_img, cur_pts[i], 5, colors[i], -1);
        // }

        // KITTI CAMERA
        // double FOCAL_LENGTH = 718.8560;
        // const Point2d pp(607.1928, 185.2157);
        // NOMAD CAMERA
        double FOCAL_LENGTH = 718.8560;
        const Point2d pp(cx, cy);

        Mat Rot, Tran;
        Mat E = findEssentialMat(cur_pts, prev_pts, FOCAL_LENGTH, pp, FM_RANSAC, 0.999, 1.0);
        recoverPose(E, cur_pts, prev_pts, Rot, Tran, FOCAL_LENGTH, pp);

        if (count < 2)
        {
            cout << "initializing";
            finalR = Rot;
            finalT = Tran;
        }
        else
        {

            currT[0] = Tran.at<double>(0);
            currT[1] = Tran.at<double>(1);
            currT[2] = Tran.at<double>(2);
            double abs_scale = 0.15; // TODO ~ distancia entre poses

            // if (abs_scale > 0.1 && (fabs(currT[2] > fabs(currT[0]))) && (fabs(currT[2]) > fabs(currT[1])))
            {
                finalT = finalT + abs_scale * finalR * Tran;
                finalR = Rot * finalR;
            }
        }

        cout << finalT << endl;
        // cout << finalR << endl;
        Vec3f euler = rotationMatrixToEulerAngles(finalR);
        cout << euler << endl;

        // const Mat CameraMatrix = Mat(3, 3, CV_64FC1, camera_param);

        // Mat EMatrix = findEssentialMat(un_prev_pts, un_cur_pts, CameraMatrix, FM_RANSAC, 0.99, 1.0);
        // Mat R, T;
        // recoverPose(EMatrix, un_prev_pts, un_cur_pts, CameraMatrix, R, T);

        // Eigen::Vector3d currT(T.at<double>(0), T.at<double>(1), T.at<double>(2));
        // double scale_ = (currT - prevT).norm();

        // if (scale_ > 0.1 && T.at<double>(0) > T.at<double>(1) && T.at<double>(0) > T.at<double>(2))
        // {
        //     finalT = finalT + scale_ * R * T;
        //     finalR = finalR * R;
        // }

        // cout << "Rot: " << finalR << "\n T:" << finalT << endl;
        Mat img;
        add(cur_img, mask, img);
        imshow("Frame", img);
        if (waitKey(1) == 27)
            break; // stop capturing by pressing ESC

        // TODO heuristica para selecionar novos pontos ?
        std::cout << "total features: " << cur_pts.size();
        if (cur_pts.size() < MAX_FEATURES)
        {
            // goodFeaturesToTrack(cur_img, new_pts, MAX_FEATURES - cur_pts.size(), 0.05, 40, Mat());
            fast->detect(cur_img, keypts);

            int i = 0;
            cur_pts.resize(keypts.size());
            for (const auto &pt : keypts)
            {
                cur_pts[i] = pt.pt;
                i++;
            }
        }

        count++;
        prev_img = cur_img;
        prev_pts = cur_pts;
        // mask = Mat::zeros(cur_img.size(), cur_img.type());
        usleep(10000);
    }

    // std::vector<KeyPoint> keypoints;
    // fast->detect(img,keypoints);

    // std::cout << "Detected: " << keypoints.size() << "keypoints\n";
    // Mat img2;

    // drawKeypoints(img,keypoints,img2);

    // imshow("window",img);
    // imshow("window2",img2);
    // int k = waitKey(0);
}

Vec3f rotationMatrixToEulerAngles(const Mat &R)
{

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
}