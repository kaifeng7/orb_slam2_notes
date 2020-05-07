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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId = 0;

/**
 * @brief Construct a new Key Frame:: Key Frame object
 * 
 * @param F Frame
 * @param pMap global Map
 * @param pKFDB KeyFrame DataBase
 */
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB) : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
                                                                   mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
                                                                   mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
                                                                   mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
                                                                   fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
                                                                   mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
                                                                   mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
                                                                   mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
                                                                   mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
                                                                   mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
                                                                   mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
                                                                   mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
                                                                   mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb / 2), mpMap(pMap)
{
    mnId = nNextId++; //将下一帧的Id赋值给当前Id，自增1

    mGrid.resize(mnGridCols); //根据栅格列数，重置栅格size
    for (int i = 0; i < mnGridCols; i++)
    {
        mGrid[i].resize(mnGridRows); //根据栅格行数，重置栅格size
        for (int j = 0; j < mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j]; //将当前帧的栅格信息拷贝给关键帧类内变量
    }

    SetPose(F.mTcw); //将当前帧的位姿拷贝给关键帧类内变量
}

/**
 * @brief 计算词袋向量
 * 
 */
void KeyFrame::ComputeBoW()
{
    if (mBowVec.empty() || mFeatVec.empty()) //判断词袋向量和特征向量是否为空
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors); //将描述子转换为描述子向量
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

/**
 * @brief set pose
 * 
 * @param Tcw_ transform matrix
 */
void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    /*--------------------------------
    T_cw=[R_cw  t_cw
          0^T   I]
    
    T_wc=[R_cw^(-1) -R_cw^(-1)*t
          0^T       I]

    ----------------------------------*/

    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3); //rotation matrix
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);         //translation matrix
    cv::Mat Rwc = Rcw.t();                           //inverse of rotation matrix
    Ow = -Rwc * tcw;                                 //center of camera

    Twc = cv::Mat::eye(4, 4, Tcw.type());
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(Twc.rowRange(0, 3).col(3));
    cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
    Cw = Twc * center;
}

/**
 * @brief get pose
 * 
 * @return cv::Mat Tcw
 */
cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

/**
 * @brief get inverse of pose 
 * 
 * @return cv::Mat Twc
 */
cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

/**
 * @brief 获取相机中心信息
 * 
 * @return cv::Mat Ow
 */
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

/**
 * @brief 获取双目相机中心信息
 * 
 * @return cv::Mat Cw
 */
cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

/**
 * @brief 获取旋转矩阵
 * 
 * @return cv::Mat Rcw
 */
cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}

/**
 * @brief 获取平移矩阵
 * 
 * @return cv::Mat tcw
 */
cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0, 3).col(3).clone();
}

/**
 * @brief 在covisibility graph 中增加连接，更新essential graph
 * 
 * @param pKF 具有共视关系的关键帧
 * @param weight 与pKF共视的MapPoint数量
 */
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (!mConnectedKeyFrameWeights.count(pKF)) //是否在已经连接的关键帧中
            mConnectedKeyFrameWeights[pKF] = weight;
        else if (mConnectedKeyFrameWeights[pKF] != weight) //是否和这次的weight相同
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

/**
 * @brief 按weight排列共视 KeyFrame
 * 
 */
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int, KeyFrame *> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size()); //根据关键帧数量，设置容器大小
    for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
        vPairs.push_back(make_pair(mit->second, mit->first)); //权重在前，关键帧在后

    sort(vPairs.begin(), vPairs.end()); //按照观测数大小进行排序
    list<KeyFrame *> lKFs;              //关键帧链
    list<int> lWs;                      //权重链
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

/**
 * @brief 获取有共视关系的KeyFrame set
 * 
 * @return set<KeyFrame*> 
 */
set<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame *> s;
    for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
        s.insert(mit->first); //取出关键帧部分
    return s;
}

/**
 * @brief 获取covisbility 中与此关键帧相连的关键帧vector
 * 
 * @return vector<KeyFrame*> 
 */
vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

/**
 * @brief 获取covisibility中与此KeyFrame相连的N帧排列好的关键帧
 * 
 * @param N 
 * @return vector<KeyFrame*> 
 */
vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

/**
 * @brief 根据weight获得KeyFrame
 * 
 * @param w 权重
 * @return vector<KeyFrame*> 比weight大的KeyFrames
 */
vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame *>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);
    if (it == mvOrderedWeights.end())
        return vector<KeyFrame *>();
    else
    {
        int n = it - mvOrderedWeights.begin();
        return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
    }
}

/**
 * @brief 获取权重
 * 
 * @param pKF KeyFrame
 * @return int 返回map的映射值
 */
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

/**
 * @brief 添加MapPoint
 * 
 * @param pMP MapPoint
 * @param idx 索引
 */
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

/**
 * @brief 删除MapPoint的匹配关系（按索引）
 * 
 * @param idx 索引
 */
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

/**
 * @brief 删除MapPoint的匹配关系（按MapPoint）
 * 
 * @param pMP MapPoint
 */
void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this); //获取MapPoint在当前KeyFrame下的索引
    if (idx >= 0)
        mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
}

/**
 * @brief 替换MapPoint的匹配关系
 * 
 * @param idx 索引
 * @param pMP MapPoint
 */
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
{
    mvpMapPoints[idx] = pMP;
}

/**
 * @brief 获取所有好的MapPoint set
 * 
 * @return set<MapPoint*> 
 */
set<MapPoint *> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint *> s;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++)
    {
        if (!mvpMapPoints[i])
            continue;
        MapPoint *pMP = mvpMapPoints[i];
        if (!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

/**
 * @brief 此KeyFrame跟踪到MapPoint的数量
 * 
 * @param minObs 最少的观测次数
 * @return int 被跟踪的点数
 */
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints = 0;                   //被跟踪的点数
    const bool bCheckObs = minObs > 0; //观测大于0，代表被观测
    for (int i = 0; i < N; i++)
    {
        MapPoint *pMP = mvpMapPoints[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                if (bCheckObs)
                {
                    if (mvpMapPoints[i]->Observations() >= minObs)
                        nPoints++;
                }
                else //没有被观测到的话，就直接算入
                    nPoints++;
            }
        }
    }

    return nPoints;
}

/**
 * @brief 获取所有和此KeyFrame有关的MapPoints
 * 
 * @return vector<MapPoint*> 
 */
vector<MapPoint *> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

/**
 * @brief 根据索引值返回MapPoint
 * 
 * @param idx 索引
 * @return MapPoint* 
 */
MapPoint *KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/**
 * @brief 更新连接关系
 *      1. 首先获得该关键帧的所有MapPoint点，统计观测到这些3d点的每个关键与其它所有关键帧之间的共视程度
 *    对每一个找到的关键帧，建立一条边，边的权重是该关键帧与当前关键帧公共3d点的个数。
 *      2. 并且该权重必须大于一个阈值，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 *      3. 对这些连接按照权重从大到小进行排序，以方便将来的处理
 *    更新完covisibility图之后，如果没有初始化过，则初始化为连接权重最大的边（与其它关键帧共视程度最高的那个关键帧），类似于最大生成树
 * 
 */
void KeyFrame::UpdateConnections()
{
    map<KeyFrame *, int> KFcounter; //<关键帧，观测到和当前帧相同MapPoint的个数>

    vector<MapPoint *> vpMP;//该KeyFrame对应的MapPoints

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for (vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)//取出所有MapPoint
    {
        MapPoint *pMP = *vit;

        if (!pMP)
            continue;

        if (pMP->isBad())
            continue;

        map<KeyFrame *, size_t> observations = pMP->GetObservations();//取出能观测到该MapPoint的所有特征点

        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            if (mit->first->mnId == mnId) //排除帧号与当前Id相同的关键帧，自己和自己不算共视
                continue;
            KFcounter[mit->first]++; //还有多少其他帧，也包含这个MapPoint
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax = 0;
    KeyFrame *pKFmax = NULL;
    int th = 15;

    vector<pair<int, KeyFrame *> > vPairs;//将权重写在前面，方便排序
    vPairs.reserve(KFcounter.size());
    for (map<KeyFrame *, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
    {
        if (mit->second > nmax)//得到共视程度最好的帧
        {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th) //weight大于阈值
        {
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->AddConnection(this, mit->second);
        }
    }

    if (vPairs.empty()) //如果没有大于阈值的KeyFrame，就将观测到相同MapPoint最多的帧添加连接
    {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(this, nmax);
    }

    sort(vPairs.begin(), vPairs.end()); //按照观测次数多少进行排序，只保存大于阈值的共视帧
    list<KeyFrame *> lKFs;
    list<int> lWs;
    for (size_t i = 0; i < vPairs.size(); i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        //mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && mnId != 0) //如果之前没有连接帧且当前帧不是第一帧时
        {
            mpParent = mvpOrderedConnectedKeyFrames.front(); //第一帧为父节点，即共视程度最高的关键帧
            mpParent->AddChild(this);                        //增加子节点，双向关系
            mbFirstConnection = false;
        }
    }
}

/**
 * @brief 添加子节点
 * 
 * @param pKF 关键帧
 */
void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

/**
 * @brief 删除子节点
 * 
 * @param pKF 待删除关键帧
 */
void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

/**
 * @brief 改变父节点 
 * 
 * @param pKF 关键帧
 */
void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

/**
 * @brief 获取子节点
 * 
 * @return set<KeyFrame*> 返回所有子节点
 */
set<KeyFrame *> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

/**
 * @brief 获取父节点
 * 
 * @return KeyFrame* 返回父节点
 */
KeyFrame *KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

/**
 * @brief 关键帧是否是子节点
 * 
 * @param pKF 关键帧
 * @return true 
 * @return false 
 */
bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

/**
 * @brief 添加回环的边
 * 
 * @param pKF 关键帧
 */
void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true; //将不删除的标志位为true
    mspLoopEdges.insert(pKF);
}

/**
 * @brief 获取回环的边
 * 
 * @return set<KeyFrame*> 
 */
set<KeyFrame *> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

/**
 * @brief 设置不删除的flag
 * 
 */
void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

/**
 * @brief 设置删除flag
 * 
 */
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mspLoopEdges.empty()) //判断回环边组中是否有关键帧
        {
            mbNotErase = false; //如果没有，可以删除
        }
    }

    if (mbToBeErased) //将要删除
    {
        SetBadFlag();
    }
}

/**
 * @brief if the KeyFrame is bad
 * first,erase its connection and observation
 * second,update the tree and the KeyFrame which is connected with it 
 * last,erase its MapPoints and Database
 */
void KeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mnId == 0) //ensure it is not first frame
            return;
        else if (mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
        mit->first->EraseConnection(this);

    for (size_t i = 0; i < mvpMapPoints.size(); i++)
        if (mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame *> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while (!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame *pC;
            KeyFrame *pP;

            for (set<KeyFrame *>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
            {
                KeyFrame *pKF = *sit;//children KeyFrame
                if (pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
                {
                    for (set<KeyFrame *>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
                    {
                        if (vpConnected[i]->mnId == (*spcit)->mnId)//if children's connected KeyFrame exists in parent candidates
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w > max)//get the KeyFrame which is the best parent candidates in children's connected KeyFrame
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if (bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);//this KeyFrame has found his parent
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if (!mspChildrens.empty())
            for (set<KeyFrame *>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw * mpParent->GetPoseInverse();
        mbBad = true;
    }

    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

/**
 * @brief get bag's flag
 * 
 * @return true 
 * @return false 
 */
bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

/**
 * @brief erase connection
 * 
 * @param pKF 关键帧
 */
void KeyFrame::EraseConnection(KeyFrame *pKF)
{
    bool bUpdate = false; // if connection has been erased，please update
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestCovisibles();
}

/**
 * @brief 获取某区域内的特征点
 * 
 * @param x center points x
 * @param y center points y
 * @param r window size
 * @return vector<size_t> all features's index in area
 */
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

/**
 * @brief 判断点是否在像素平面内
 * 
 * @param x 坐标x
 * @param y 坐标y
 * @return true 
 * @return false 
 */
bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

/**
 * @brief 双目摄像头的反投影（必须要有深度值信息）
 * 
 * @param i 索引
 * @return cv::Mat MapPoint in world coordinate
 */
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i]; //depth
    if (z > 0)
    {
        const float u = mvKeys[i].pt.x; //camera coordinate
        const float v = mvKeys[i].pt.y;
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z); //相机坐标系下的坐标

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3); //变换到世界坐标系下的坐标
    }
    else
        return cv::Mat();
}

/**
 * @brief 计算MapPoints集合在此帧深度的中位数
 * 
 * @param q 
 * @return float 
 */
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint *> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3); //旋转矩阵z部分的值
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2, 3); //平移矩阵z部分的值
    for (int i = 0; i < N; i++)
    {
        if (mvpMapPoints[i])
        {
            MapPoint *pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw; //转换到当前坐标系下
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end()); //每个深度值排序

    return vDepths[(vDepths.size() - 1) / q];
}

} // namespace ORB_SLAM2
