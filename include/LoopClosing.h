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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:

    typedef pair<set<KeyFrame*>,int> ConsistentGroup;    
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    //将关键帧加入待回环检测队列
    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    //检测队列中是否有新的关键帧
    bool CheckNewKeyFrames();

    //检测是否产生回环
    bool DetectLoop();

    //相似变换矩阵是否计算完成
    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    //回环校正
    void CorrectLoop();

    //复位请求
    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;
    
    //回环检测是否结束
    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;//待回环检测的关键帧队列

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;//当前关键帧
    KeyFrame* mpMatchedKF;//找到和mpCurrentKF形成闭环关键帧
    std::vector<ConsistentGroup> mvConsistentGroups;//所有连续的关键帧组
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;//充分连接的候选关键帧组(一致性超过阈值的帧)
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;//将mpMatchedKF闭环关键帧相连的关键帧全部取出来放入
    std::vector<MapPoint*> mvpCurrentMatchedPoints;//将mvpLoopMapPoints投影到当前关键帧mpCurrentKF进行投影得到的匹配,判断是否冲突
    std::vector<MapPoint*> mvpLoopMapPoints;//mvpCurrentConnectedKFs的MapPoint
    cv::Mat mScw;//通过sim3 计算的当前帧到world的变换
    g2o::Sim3 mg2oScw;

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;


    bool mnFullBAIdx;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
