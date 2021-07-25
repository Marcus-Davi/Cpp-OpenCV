#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <unistd.h>
#define FRAMES_PATH "/home/marcus/Workspace/Coding/Python/bm_frames/"

using namespace cv;
using namespace std;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < 1280 - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < 720 - BORDER_SIZE;
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

        // Project points
        


        count++;
        prev_img = cur_img;
        prev_pts = cur_pts;
        mask = Mat::zeros(cur_img.size(), cur_img.type());
        usleep(10000);
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