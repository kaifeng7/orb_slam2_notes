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

#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
//定义一帧中有多少个图像网格
#define FRAME_GRID_ROWS 48 //网格的行数
#define FRAME_GRID_COLS 64 //网格的列数

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;//用于重定位的ORB特征字典

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;//去畸变参数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    // Stereo baseline multiplied by fx.
    float mbf;//baseline*fx

    // Stereo baseline in meters.
    float mb;//baseline

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;//远点和近点的深度阈值

    // Number of KeyPoints.
    int N;//keyPoints数量

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // 校正的操作在帧的构造函数中进行
    std::vector<cv::KeyPoint> mvKeys;//原始左图像提取出的特征点(未校正)
    std::vector<cv::KeyPoint>mvKeysRight;//原始右图像提取出的特征点(未校正)
    std::vector<cv::KeyPoint> mvKeysUn;//校正mvKeys后的特征点

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;//左目像素点在右目中的对应点的横坐标(因为纵坐标是一致的)
    std::vector<float> mvDepth;//the depth of feature

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;//和词袋模型有关的向量
    DBoW2::FeatureVector mFeatVec;//和词袋模型中特征有关的向量

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors;//左目相机中特征点对应的描述子
    cv::Mat mDescriptorsRight;//右目相机中特征点对应的描述子

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;//每个特征点对应的MapPoint，如果特征点没有对应的地图点，那么存储一个空指针

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;//观测不到的3d点，在Optimizer::PoseOptimization使用

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;//可以确定在哪个grid里
    static float mfGridElementHeightInv;//可以确定在哪个grid里
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];//每个图像网格内特征点的Id

    cv::Mat mTcw;// Camera pose.


    static long unsigned int nNextId;//Next Frame id.
                                     //在整个系统开始执行的时候被初始化，在全局区被初始化
    long unsigned int mnId;//Current Frame id

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;//指向参考关键帧

    // Scale pyramid info.
    int mnScaleLevels; //图像金字塔的层数
    float mfScaleFactor; //图像金字塔的尺度因子
    float mfLogScaleFactor;//log(scaleFactor)
    vector<float> mvScaleFactors;//每一层的缩放因子
    vector<float> mvInvScaleFactors;// 1/ScaleFactor
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    // grid的边界
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    //由于第一帧以及slam系统进行重新校正后的第一帧会有一些特殊的初始化处理操作
    static bool mbInitialComputations;//标记静态成员变量是否需要被赋值
    


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw;//rotation Matrix
    cv::Mat mtcw;//translation Matrix
    cv::Mat mRwc;//inverse of rotation Matrix
    cv::Mat mOw; //==mtwc,camera center
};

}// namespace ORB_SLAM

#endif // FRAME_H
