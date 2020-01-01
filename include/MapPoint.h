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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();
    
    //得到观测map
    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    //在地图点中添加关键帧 表明该地图点属于哪一个关键帧
    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    //该地图点是KeyFrame中的第几个观测点，如果没有，返回-1
    int GetIndexInKeyFrame(KeyFrame* pKF);
    //该地图点是否被KeyFrame观测到
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    //增加查找CurrentFrame检查到的MapPoints的次数
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    //从众多观测到该MapPoint的特征点中挑选出分数最高的descriptor
    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();
    
    //更新该MapPoint平均观测方向以及观测距离的范围
    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;//the current MapPoint id
    static long unsigned int nNextId;//the next MapPoint id
    long int mnFirstKFid;//the first KeyFrame id
    long int mnFirstFrame;//the first Frame
    int nObs;//number of observation

    // Variables used by the tracking
    float mTrackProjX;//the x of project when tracking
    float mTrackProjY;//the y of project when tracking
    float mTrackProjXR;// the radius of search
    bool mbTrackInView;//flag to identify the point in view
    int mnTrackScaleLevel;//scale level when tracking
    float mTrackViewCos;//the cos of view when tracking
    long unsigned int mnTrackReferenceForFrame;//id of reference frame
    long unsigned int mnLastFrameSeen;//number of MapPoints viewed in last frame 

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;// local BA for KeyFrame
    long unsigned int mnFuseCandidateForKF;// 

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;//pose of global BA
    long unsigned int mnBAGlobalForKF;//global BA for KeyFrame


    static std::mutex mGlobalMutex;// global mutex

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;//pose in world coordinate

     // Keyframes observing the point and associated index in keyframe
     //该特征点在哪个KeyFrame中被观测到，以及是KeyFrame中的第几个特征点
     std::map<KeyFrame*,size_t> mObservations;

     // Mean viewing direction
     cv::Mat mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;//number of visible KeyFrame
     int mnFound;//number of Found KeyFrame

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;//used by replacing MapPoint

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;//global map

     std::mutex mMutexPos;//MapPoint Pose Mutex
     std::mutex mMutexFeatures;//Features Mutex
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
