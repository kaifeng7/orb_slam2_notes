/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"



namespace ORB_SLAM2 
{

class Sim3Solver
{
public:

    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float GetEstimatedScale();


protected:

    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

    void CheckInliers();

    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches
    KeyFrame* mpKF1;
    KeyFrame* mpKF2;

    std::vector<cv::Mat> mvX3Dc1;//camera下的坐标
    std::vector<cv::Mat> mvX3Dc2;//camera下的坐标
    std::vector<MapPoint*> mvpMapPoints1;//存放匹配的MapPoints1
    std::vector<MapPoint*> mvpMapPoints2;//存放匹配的MapPoints2
    std::vector<MapPoint*> mvpMatches12;
    std::vector<size_t> mvnIndices1;
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    std::vector<size_t> mvnMaxError1;//重投影误差阈值1
    std::vector<size_t> mvnMaxError2;//重投影误差阈值2

    int N;
    int mN1;//pKF2特征点的个数

    // Current Estimation
    cv::Mat mR12i;//当前 旋转
    cv::Mat mt12i;//当前 平移
    float ms12i;//当前 Scale
    cv::Mat mT12i;//当前 Sim3 变换
    cv::Mat mT21i;//当前 Sim3 变换
    std::vector<bool> mvbInliersi;//内点集合
    int mnInliersi;//内点数

    // Current Ransac State
    int mnIterations;//迭代次数
    std::vector<bool> mvbBestInliers;//最好情况下的内点集合
    int mnBestInliers;//最好情况下的内点数
    cv::Mat mBestT12;//最好的Sim3
    cv::Mat mBestRotation;//最好的旋转
    cv::Mat mBestTranslation;//最好的平移
    float mBestScale;//最好的尺度

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    // Projections
    std::vector<cv::Mat> mvP1im1;
    std::vector<cv::Mat> mvP2im2;

    // RANSAC probability
    double mRansacProb;//ransac 可接受概率

    // RANSAC min inliers
    int mRansacMinInliers;//最小可接受的内点数

    // RANSAC max iterations
    int mRansacMaxIts;//最大迭代次数

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1;//内参1
    cv::Mat mK2;//内参2

};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
