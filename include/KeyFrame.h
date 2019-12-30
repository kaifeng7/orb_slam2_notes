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
    //设置位姿
    void SetPose(const cv::Mat &Tcw);
    //获取位姿
    cv::Mat GetPose();
    //获取位姿的逆
    cv::Mat GetPoseInverse();
    //获取传感器的中心坐标
    cv::Mat GetCameraCenter();
    //获取双目的中心坐标
    cv::Mat GetStereoCenter();
    //获取旋转矩阵
    cv::Mat GetRotation();
    //获取平移矩阵
    cv::Mat GetTranslation();

    // Bag of Words Representation
    //计算词袋特征
    void ComputeBoW();

    // Covisibility graph functions
    //增加连接
    void AddConnection(KeyFrame* pKF, const int &weight);
    //移除连接
    void EraseConnection(KeyFrame* pKF);
    //更新连接
    void UpdateConnections();
    //更新最好的共视
    void UpdateBestCovisibles();
    //获取连接的关键帧
    std::set<KeyFrame *> GetConnectedKeyFrames();
    //得到共视关键帧
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    //得到共视度最高的关键帧
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    //通过权重来获得Covisibility
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    //获取权重
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    //增加子树
    void AddChild(KeyFrame* pKF);
    //移除子树
    void EraseChild(KeyFrame* pKF);
    //改变父节点
    void ChangeParent(KeyFrame* pKF);
    //获取子节点
    std::set<KeyFrame*> GetChilds();
    //获得父节点
    KeyFrame* GetParent();
    //判断此关键帧是否有子节点
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    //增加回环的边
    void AddLoopEdge(KeyFrame* pKF);
    //获取回环的边
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    
    //关键帧中添加地图点，说明在该关键帧下可以看到哪个地图点
    void AddMapPoint(MapPoint* pMP, const size_t &idx);

    //移除关键帧中匹配到的地图点（输入为索引）
    void EraseMapPointMatch(const size_t &idx);
    
    //移除关键帧中匹配到的地图点（输入为MapPoint）
    void EraseMapPointMatch(MapPoint* pMP);
    //替换MapPoint的匹配
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    //获取MapPoints组
    std::set<MapPoint*> GetMapPoints();
    //获取MapPoint的匹配
    std::vector<MapPoint*> GetMapPointMatches();
    //可观察到的地图点
    int TrackedMapPoints(const int &minObs);
    //获取MapPoint
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    //获取区域内的特征点
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    //重投影双目摄像头
    cv::Mat UnprojectStereo(int i);

    // Image
    //判断是否在图像里
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    
    //设置关键帧为不可移除状态
    void SetNotErase();
    //设置关键帧为可移除状态
    void SetErase();

    // Set/check bad flag
    //设置坏的标志位
    void SetBadFlag();
    //判断是否是坏的KeyFrame
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    //计算深度信息（只在单目模式下）
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

    //下一帧ID
    static long unsigned int nNextId;
    //当前ID
    long unsigned int mnId;
    //当前帧ID
    const long unsigned int mnFrameId;

    //时间戳
    const double mTimeStamp;

    // Grid (to speed up feature matching)
    //用于加速特征点匹配部分

    //栅格的长
    const int mnGridCols;
    //栅格的宽
    const int mnGridRows;
    //栅格的长的倒数
    const float mfGridElementWidthInv;
    //栅格的宽的倒数
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    //用在跟踪部分

    //跟踪参考帧
    long unsigned int mnTrackReferenceForFrame;
    //关键帧中使用的目标点
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    //用在局部地图部分

    //关键帧的局部BA
    long unsigned int mnBALocalForKF;
    //关键帧的固定点的BA
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    //用在关键帧数据集部分

    //回环队列
    long unsigned int mnLoopQuery;
    //回环字符标志
    int mnLoopWords;
    //回环得分
    float mLoopScore;
    //重定位的队列
    long unsigned int mnRelocQuery;
    //重定位的字符标志
    int mnRelocWords;
    //重定位的得分
    float mRelocScore;

    // Variables used by loop closing
    //用于闭环检测部分

    //全局BA的变换矩阵
    cv::Mat mTcwGBA;
    //之前全局BA的变换矩阵
    cv::Mat mTcwBefGBA;
    //关键帧用于全局BA
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    //相机标定
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    //关键点数量
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)

    //存放关键点组
    const std::vector<cv::KeyPoint> mvKeys;
    //存放去畸变后关键点组
    const std::vector<cv::KeyPoint> mvKeysUn;
    //存放右图上的值组
    const std::vector<float> mvuRight; // negative value for monocular points
    //存放深度值组
    const std::vector<float> mvDepth; // negative value for monocular points
    //描述子
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;   //词袋向量

    DBoW2::FeatureVector mFeatVec;  //特征向量


    // Pose relative to parent (this is computed when bad flag is activated)
    //相对于父节点的变换矩阵
    cv::Mat mTcp;

    // Scale
    //尺度层数
    const int mnScaleLevels;
    //尺度因子
    const float mfScaleFactor;
    //尺度因子取对数
    const float mfLogScaleFactor;
    //尺度因子组
    const std::vector<float> mvScaleFactors;
    //平方组
    const std::vector<float> mvLevelSigma2;
    //平方倒数组
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    //图像边界值
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
   
    const cv::Mat mK; //内参矩阵


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;//世界到相机的变换矩阵
    cv::Mat Twc;//相机到世界的变换矩阵
    cv::Mat Ow;//=-R^T*t,相机在世界坐标系下的坐标

    //Stereo middel point. Only for visualization
    cv::Mat Cw; //双目摄像机baseline中点坐标

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;//在此KeyFrame下，可观测到的MapPoints

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;//KeyFrame数据集指针
    ORBVocabulary* mpORBvocabulary;//orb词汇指针

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;//二位容器的栅格

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;//连接关键帧与权重的map
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;//按顺序连接的关键帧组
    std::vector<int> mvOrderedWeights;//顺序权重

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;//第一个是否连接
    KeyFrame* mpParent;//关键帧的父节点
    std::set<KeyFrame*> mspChildrens;//子节点组
    std::set<KeyFrame*> mspLoopEdges;//闭环边组

    // Bad flags
    bool mbNotErase;//判断不要移除
    bool mbToBeErased;//判断将要移除
    bool mbBad;//判断坏点

    // Only for visualization
    float mHalfBaseline; // 基线距离的一半

    Map* mpMap;//Map

    std::mutex mMutexPose;//KeyFrame‘s Pose
    std::mutex mMutexConnections;//KeyFrame connections
    std::mutex mMutexFeatures;//KeyFrame and Features
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
