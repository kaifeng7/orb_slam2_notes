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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include <unistd.h>
#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,//当前没有图片，收到新的图片后，转为NOT_INITIALIZED
        NOT_INITIALIZED=1,//当前没有初始化追踪线程，初始化后，转为OK
        OK=2,//没有丢帧或者复位的情况下系统一直处于OK状态
        LOST=3//当前追踪线程丢失，上一帧追踪失败，下一帧进行重定位
    };

    eTrackingState mState;//跟踪状态标志
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame，当前帧
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;//之间匹配的特征点
    std::vector<cv::Point2f> mvbPrevMatched;//前一帧特征点
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;//每一图像帧与其参考帧之间的姿态变换关系（用于绘制轨迹）
    list<KeyFrame*> mlpReferences;//每一图像帧的参考关键帧（用于绘制轨迹）
    list<double> mlFrameTimes;//每一图像帧的时间戳（用于绘制轨迹）
    list<bool> mlbLost;//每一图像帧的跟踪状态

    // True if local mapping is deactivated and we are performing only localization
    //纯定位模式，true->只定位，不建图局部地图
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    // 真正的跟踪流程
    void Track();

    // Map initialization for stereo and RGB-D
    // 双目摄像头和RGBD摄像头的地图初始化
    void StereoInitialization();

    // Map initialization for monocular
    // 单目摄像头的地图初始化
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    //检查并更新LastFrame中的MapPoints
    void CheckReplacedInLastFrame();

    //参考关键帧进行跟踪
    //1.首先计算当前帧的词袋向量。
    //2.根据词袋向量进行参考关键帧和当前帧进行匹配 ，得到匹配点。
    //3.如果匹配点的数量满足要求，则将匹配点设为当前帧的地图点，上一帧的位姿设为当前帧的位姿。优化过后剔除外点。
    //4.最后根据匹配点的数量判定跟踪是否成功。
    bool TrackReferenceKeyFrame();
    
    void UpdateLastFrame();

    bool TrackWithMotionModel();


    bool Relocalization();

    //更新局部地图
    void UpdateLocalMap();

    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    //局部地图跟踪
    //1.更新局部地图。将之前的局部地图数据清空，重新构建局部地图。构建局部地图的过程就是重新确定局部地图关键帧和局部地图地图点的过程。
    //2.局部地图中的地图点与当前帧的地图点进行匹配，然后利用匹配的地图点来优化当前帧的位姿。
    //3.根据优化后的位姿重新更新匹配点的内点和外点。
    //4.根据内点数量判定跟踪是否成功。
    bool TrackLocalMap();

    //匹配局部地图中的地图点和当前帧的地图点
    void SearchLocalPoints();

    //插入新的关键帧
    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    // 只在纯定位模式下才被使用
    // false->匹配了很多MapPoints，跟踪正常
    // true ->匹配了很少MapPoints，跟踪失败
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;//参考关键帧
    std::vector<KeyFrame*> mvpLocalKeyFrames;//局部地图的关键帧
    std::vector<MapPoint*> mvpLocalMapPoints;//局部地图的地图点
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;//整个地图

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;//上一关键帧
    Frame mLastFrame;//上一帧
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;//上一次重定位的那一帧

    //Motion Model
    //上一帧与上上帧之间的位姿变换关系
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
