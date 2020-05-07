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

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>

#include <mutex>

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) : mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
                                                                                                                                                                                              mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
                                                                                                                                                                                              mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    // K = |fx 0 cx|
    //     |0 fy cy|
    //     |0  0  1|

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl
         << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];//每一帧提取特征点数 1000
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];//图像建立金字塔时的变化尺度 1.2
    int nLevels = fSettings["ORBextractor.nLevels"];// 尺度金字塔的层数 8
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];//提取fast特征点的默认阈值 20
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];//如果默认阈值提取不到这么多fast时，使用最小阈值8

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::MONOCULAR)//单目需要进行初始化
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;//判断3D点的远近
        cout << endl
             << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    //1.将RGB图像转化为灰度图像
    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
        }
    }
    //2.构造Frame
    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    //3.跟踪
    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)//如果没有初始化
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

/**
 * @brief Main tracking function
 * 
 */
void Tracking::Track()
{
    if (mState == NO_IMAGES_YET)//图像复位过或者第一次运行
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED) //没有初始化
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if (mState != OK)
            return;
    }
    else //初始化完毕
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking) //不是纯追踪模式
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if (mState == OK) //追踪正常
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                //优先选择根据运动跟踪
                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) //刚完成重定位
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if (!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else //追踪丢失
            {
                bOK = Relocalization(); //重定位
            }
        }
        else //纯追踪模式
        {
            // Localization Mode: Local Mapping is deactivated

            if (mState == LOST) //追踪丢失
            {
                bOK = Relocalization();
            }
            else //追踪正常
            {
                if (!mbVO) //足够的MapPoints匹配
                {
                    // In last frame we tracked enough MapPoints in the map

                    if (!mVelocity.empty()) //速度不为零
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else //很少的MapPoints匹配
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    // 运动模型跟踪与重定位同时进行,更相信重定位

                    bool bOKMM = false;    //运动跟踪是否成功
                    bool bOKReloc = false; //重定位是否成功
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty()) //如果速度不为空
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization(); //重定位获取相机位资

                    if (bOKMM && !bOKReloc) //如果跟踪成功，重定位失败
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO) //此帧匹配了很少的MapPoints，跟踪效果不好
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc) //重定位成功
                    {
                        mbVO = false; //重定位成功，就可以work，跟踪与重定位更相信重定位
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // 将新的关键帧作为 referenceKF
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!mbOnlyTracking)//不是纯跟踪模式
        {
            if (bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.

            //帧间匹配得到初始姿态后，对local map进行跟踪得到更多的匹配，并优化当前位姿
            if (bOK && !mbVO)//重定位成功
                bOK = TrackLocalMap();
        }

        if (bOK)
            mState = OK;
        else
            mState = LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            //清除UpdateLastFrame中为当前帧临时添加的MapPoints(stereo)
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            // 检测并插入关键帧，
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (int i = 0; i < mCurrentFrame.N; i++)//删除BA中检测的outlier
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)//跟踪失败
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);//保存上一帧数据
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // 记录位姿信息，用于轨迹复现
    // 通过当前帧的位姿是否为空来判断是否跟踪成功
    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse(); //当前帧的参考帧到当前帧的位姿变换矩阵
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else//如果跟踪失败，相对位姿使用上一次的值
    {
        // This can happen if tracking is lost
        // 均相当于在链表末端复制了最后一个元素
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

/**
 * @brief stereo初始化，有深度可以单帧初始化
 * 
 */
void Tracking::StereoInitialization()
{
    if (mCurrentFrame.N > 500) //当前帧关键点的数量大于500，才将此帧作为初始帧并认为其为关键帧
    {
        // Set Frame pose to the origin
        // 1.设置初始帧的位资
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Create KeyFrame
        // 2.将当前帧（第一帧）作为初始关键帧（调用关键帧的构造函数）
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        //3.将关键帧加入地图中
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        //4.创建地图点并将其与关键帧建立联系
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i]; //当前帧的第i个关键点的深度值
            if (z > 0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);//反投影得到3d坐标
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap); //用该特征点构建新的MapPoint
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP; //将MapPoint加入当前帧的MapPoints中，为当前Frame与MapPoint之间建立联系
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame); //更新上一帧关键帧
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);          //将初始KeyFrame加入到局部地图中
        mvpLocalMapPoints = mpMap->GetAllMapPoints(); //将全部MapPoint加入到当前局部地图点
        mpReferenceKF = pKFini;                       //将当前关键帧作为参考关键帧
        mCurrentFrame.mpReferenceKF = pKFini;         //将当前关键帧作为当前帧的参考关键帧

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini); //将KeyFrame加入Map的原始关键帧

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK; //更新跟踪状态
    }
}

/**
 * @brief 单目地图初始化
 * 
 */
void Tracking::MonocularInitialization()
{

    if (!mpInitializer)//创建单目初始器
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)// 单目初始帧提取的特征点数必须大于100，否则放弃该帧图像
        {
            //1.得到用于初始化的第一帧
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        //2.单目初始化第二帧
        // 前两帧特征点数必须大于100
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        // 3.寻找匹配点
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        // 4.匹配点数必须大于100
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            return;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        //5.通过H和F模型进行单目初始化，得到两帧间的相对运动，初始化MapPoints
        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F)); // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

/**
 * @brief Monocular 初始化
 * 
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    //1.初始关键帧和当前关键帧的描述子转为Bow
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    //2.将关键帧插入地图点
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    //3.将3d点转为MapPoint
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);//构造MapPoint

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);//观测到该MapPoint的KeyFrame
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();//描述子
        pMP->UpdateNormalAndDepth();//平均观测方向和深度范围

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    // 4.更新关键帧间的连接关系，对于一个新创建的关键帧都会进行一次关键帧连接关系更新
    // 3d点和KeyFrame之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3d点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    // BA优化
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1
    // 6. 深度归一化为1，归一化两针之间的变换
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    //7.局部地图中添加初始关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);//将KeyFrame加入Map的原始关键帧

    mState = OK;//初始化成功
}

/**
 * @brief 上一帧中的MapPoint是否被替换
 *         在local mapping 和loop closingn 中存在fuse mapping
 * 
 */
void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 对参考帧的MapPoint进行跟踪
 * 
 * @return true 
 * @return false 
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    //1.计算CurrentFrame的词袋向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);        //匹配器
    vector<MapPoint *> vpMapPointMatches; //匹配的地图点

    //2.根据词袋向量进行参考关键帧和当前帧进行匹配，得到匹配点
    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
        return false; //匹配失败

    // 3.如果匹配点的数量满足要求，则将匹配点设为当前帧的地图点
    // 上一帧的位姿设为当前帧的位姿，并进行优化
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    //4. 剔除外点
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    //根据剔除外点后的匹配点数量判定跟踪是否成功
    return nmatchesMap >= 10;
}

/**
 * @brief stereo/rgbd
 *        通过深度信息为上一帧产生新的MapPoint
 * 
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr * pRef->GetPose());//Tlr * Trw = Tlw last->reference->world

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)//如果上一帧为关键帧
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

/**
 * @brief 根据匀速模型对上一帧的MapPoints进行跟踪
 *      1.根据上一帧的位姿和速度来计算当前帧的位姿。
 *      2.遍历上一帧中所有地图点，将上一帧的地图点向当前帧进行投影，投影过后在当前帧中找到一个描述子距离最相近的特征点作为投影点的匹配点。
 *      3.如果匹配点的数量满足要求，则对当前帧进行位姿优化。优化过后剔除外点。
 *      4.最后根据匹配点的数量判断跟踪是否成功。
 * 
 * @return true 
 * @return false 
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9, true); //匹配器

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    //1.对于stereo/rgbd 更新上一帧的相机位姿
    UpdateLastFrame();

    //2.根据上一帧的位姿和速度估计当前帧位姿
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw); 

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    int th;
    if (mSensor != System::STEREO)
        th = 15;
    else
        th = 7;
    //将上一帧中的地图点投影到当前帧并进行匹配，缩小匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

    // If few matches, uses a wider window search
    if (nmatches < 20) //匹配的特征点数小于20，增大匹配阈值并重新匹配，即扩大匹配范围
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20) //如果还是小于20则跟踪失败
        return false;

    // Optimize frame pose with all matches
    //3.位姿优化
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    //4.剔除外点MapPoint
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    if (mbOnlyTracking)
    {
        //匹配数少于10，则追踪失败
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap >= 10;
}

/**
 * @brief 对Local Map的MapPoints进行跟踪
 *      1. 更新局部地图，包括局部关键帧和关键点
 *      2. 对局部MapPoints进行投影匹配
 *      3. 根据匹配对估计当前帧的姿态
 *      4. 根据姿态剔除误匹配
 * 
 * @return true 
 * @return false 
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    //1.更新局部关键帧和局部地图点
    UpdateLocalMap();

    //2.在局部地图中查找与当前帧匹配的MapPoint
    SearchLocalPoints();

    // Optimize Pose
    //3.更新局部所有MapPoints后对位姿再次优化
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    //4.更新当前帧的MapPoint被观测程度，并统计跟踪局部地图的效果
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])//当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)//该MapPoint被其他关键帧观测到过
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    //当前帧ID<上一次重定位帧ID+最大帧数 && 内点数<50
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    if (mnMatchesInliers < 30)
        return false;
    else
        return true;
}

/**
 * @brief 判断当前帧是否为关键帧
 * 
 * @return true 
 * @return false 
 */
bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking)//纯定位模式不添加
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())//局部地图被闭环检测模块使用
        return false;

    const int nKFs = mpMap->KeyFramesInMap(); //整个地图中关键帧的数量

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    //当前帧ID<上次重定位帧ID+最大帧数&&当前关键帧数>最大帧数，是否距离上一次插入时间太短
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    //参考关键帧为UpdateLocalKeyFrame中与当前关键帧共视程度最高的关键帧
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); //参考关键帧中被观察2-3的地图点数

    // Local Mapping accept keyframes?
    // 局部建图线程处于空闲状态
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0; //当前帧中未被跟踪的近点数
    int nTrackedClose = 0;    //当前帧中被跟踪的近点
    if (mSensor != System::MONOCULAR)
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    //近点中被追踪数量<100 && 近点未被追踪点数量>70
    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    // 当前帧内点数（参考关键帧中所有地图点被观测到的次数>2或者3次的地图点数量）
    // 是否需要插入关键帧
    float thRefRatio = 0.75f;
    if (nKFs < 2)//关键帧只有一帧，则插入关键帧的阈值设置较低
        thRefRatio = 0.4f;

    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 此帧距离上次插入关键帧是否超过了最大帧数，很长时间间没有插入关键帧
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // 此帧距离上次插入关键帧是否已经超过了最小帧数 && 此时局部地图线程处于空闲状态
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    // 此帧跟踪的内点少于参考关键帧中被观测到2-3次的地图点的0.25倍||需要回环
    const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 此帧跟踪的内点少于参考关键帧中被观测到2-3次的地图点的thRefRatio倍 && 当前帧地图点中内点>15，与之前参考帧重复度不高
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

    if ((c1a || c1b || c1c) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)//队列里不能阻塞太多关键帧
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief 创建新的关键帧
 * 
 */
void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    //1.当前帧构造成关键帧
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    //2.当前帧设为参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    //3.stereo生成新的MapPoint
    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();//根据Tcw计算

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());//深度从小到大排列

            int nPoints = 0;//距离较近的点包装成MapPoint
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                    //每次创建MapPoint后都要做
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > 100)//决定了点云稠密度
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief 对局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 * 
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 当前帧的MapPoint不参与匹配，因为一定存在
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();//更细能观测到该点的帧数加1
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;//标记当前帧被观测到
                pMP->mbTrackInView = false;//标记该店被投影，因为已经匹配到
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    // 将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配  
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))// 判断LocalMapPoints中的点是否在在视野内

        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if (mSensor == System::RGBD)
            th = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)// 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

/**
 * @brief 更新局部关键帧，和局部地图点
 * 
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 更新局部地图点
 * 
 */
void Tracking::UpdateLocalPoints()
{
    //1.清空局部MapPoints
    mvpLocalMapPoints.clear();

    //2.添加局部关键帧的MapPoints
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)//防止重复添加局部MapPoint
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;//local MapPoint在tracking时的参考Frame
            }
        }
    }
}

/**
 * @brief 更新局部关键帧
 * 
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 1.遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoint的关键帧
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // 1.能观测到当前帧MapPoints的关键帧 作为局部关键帧
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 2.与1得到的局部关键帧共视程度很高得关键帧作为局部关键帧
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);//最佳共视的10帧

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChilds();//自己的子关键帧
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();//自己的父关键帧
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    //3.更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @brief  重定位的过程中进行了三次匹配三次优化。
 * 第一次匹配为词袋匹配，用于初步确定当前帧的地图点。
 * 后两次匹配均为投影匹配，目的是为了增加匹配点，为优化位姿做准备。
 * 而三次优化的过程是为了根据匹配点不断的优化当前帧的位姿，使其满足要求。
 * 
 * @return true 
 * @return false 
 */
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // 1.计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // 2.找到与当前帧相似的候选关键字
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty()) //不存在候选关键帧
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver *> vpPnPsolvers; //PnP求解器
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            //3.通过Bow进行匹配
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]); //设置PnP求解器
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            //4.通过EPNP法估计姿态
            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers); //迭代5次，计算当前帧位资

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }
                //5,位姿优化
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10) //优化后内点数<10
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // 6,如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100); //附加匹配点

                    if (nadditional + nGood >= 50) //附加匹配点和内点之和>=50
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    { //重定位成功，将上次重定位的ID设为本帧的ID
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if (mpViewer)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} // namespace ORB_SLAM2
