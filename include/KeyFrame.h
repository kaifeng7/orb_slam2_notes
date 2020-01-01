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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);
    //权重的比较
    static bool weightComp( int a, int b){
        return a>b;
    }

    //关键帧ID的比较
    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId; //下一帧ID
    long unsigned int mnId; //当前ID
    const long unsigned int mnFrameId; //当前帧ID
    const double mTimeStamp; //时间戳


    // Grid (to speed up feature matching)
    //用于加速特征点匹配部分

    const int mnGridCols; //number of grid columns
    const int mnGridRows; //number of grid rows
    const float mfGridElementWidthInv; //栅格的长的倒数
    const float mfGridElementHeightInv; //栅格的宽的倒数


    // Variables used by the tracking
    //用在跟踪部分

    long unsigned int mnTrackReferenceForFrame; //跟踪参考帧
    long unsigned int mnFuseTargetForKF; //关键帧中使用的目标点

    // Variables used by the local mapping
    //用在局部地图部分

    long unsigned int mnBALocalForKF; //关键帧的局部BA
    long unsigned int mnBAFixedForKF; //关键帧的固定点的BA

    // Variables used by the keyframe database
    //用在关键帧数据集部分

    long unsigned int mnLoopQuery; //number of loop query
    int mnLoopWords; //回环字符标志
    float mLoopScore; //score of loop 
    long unsigned int mnRelocQuery; //number of relocalization query
    int mnRelocWords; //重定位的字符标志
    float mRelocScore; //score of relocalization


    // Variables used by loop closing
    //用于闭环检测部分

    cv::Mat mTcwGBA; //transformation of global BA
    cv::Mat mTcwBefGBA; //transformation of before global BA
    long unsigned int mnBAGlobalForKF; //关键帧用于全局BA

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth; //相机标定

    // Number of KeyPoints
    const int N; //关键点数量


    // KeyPoints, stereo coordinate and descriptors (all associated by an index)

    const std::vector<cv::KeyPoint> mvKeys; //存放关键点组
    const std::vector<cv::KeyPoint> mvKeysUn; //存放去畸变后关键点组
    const std::vector<float> mvuRight; //存放右图上的值组 // negative value for monocular points
    const std::vector<float> mvDepth; //存放深度值组 // negative value for monocular points    
    const cv::Mat mDescriptors; //描述子

    //BoW
    DBoW2::BowVector mBowVec; //词袋向量
    DBoW2::FeatureVector mFeatVec; //特征向量

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp; //相对于父节点的变换矩阵

    // Scale

    const int mnScaleLevels; //尺度层数
    const float mfScaleFactor; //尺度因子
    const float mfLogScaleFactor; //尺度因子取对数
    const std::vector<float> mvScaleFactors; //尺度因子组
    const std::vector<float> mvLevelSigma2; //平方组
    const std::vector<float> mvInvLevelSigma2; //平方倒数组

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
   
    const cv::Mat mK;   //internal parameter matrix


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw; //transform matrix of world to camera 
    cv::Mat Twc; //transform matrix of camera to world 
    cv::Mat Ow; //=-R^T*t,camera center

    //Stereo middel point. Only for visualization
    cv::Mat Cw; //双目摄像机baseline中点坐标

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints; //the MapPoints which are observed

    // BoW
    KeyFrameDatabase* mpKeyFrameDB; //KeyFrame Database
    ORBVocabulary* mpORBvocabulary; //orb vocabulary

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid; //二位容器的栅格
    std::map<KeyFrame*,int> mConnectedKeyFrameWeights; //连接关键帧与权重的map
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; //按顺序连接的关键帧组
    std::vector<int> mvOrderedWeights; //顺序权重

    // Spanning Tree and Loop Edges
    bool mbFirstConnection; //the flag of first connection
    KeyFrame* mpParent; //parent of KeyFrame
    std::set<KeyFrame*> mspChildrens; //childrens of KeyFrame
    std::set<KeyFrame*> mspLoopEdges; //loop edges of KeyFrame

    // Bad flags
    bool mbNotErase; //the flag of not erase
    bool mbToBeErased; //the flag of to be erased
    bool mbBad; //the flag of bad 

    // Only for visualization
    float mHalfBaseline; //half of baseline

    Map* mpMap; //Map

    std::mutex mMutexPose; //KeyFrame‘s Pose
    std::mutex mMutexConnections; //KeyFrame connections
    std::mutex mMutexFeatures; //KeyFrame and Features
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
