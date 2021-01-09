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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>

#include "Converter.h"

#include <mutex>

namespace ORB_SLAM2
{

/**
 * @brief 将map中所有KeyFrame位姿和MapPoint位置作为优化进行BA
 *        两处用到1.单目初始化 2.闭环优化
 *  
 * @param pMap 地图类型的指针
 * @param nIterations 迭代的次数
 * @param pbStopFlag 停止标志的指针
 * @param nLoopKF 在id为nLoopKF处进行BA
 * @param bRobust 是否使用核函数
 */
void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

/**
 * @brief 通过优化vpKF的pose、vpMP等优化变量，使得vpMP通过vpKF里的位姿投影到vpKF的二维坐标的重投影误差最小
 * 1.Vertex: g2o::VertexSE3Expmap() CurrentKF的Tcw
 *           g2o::VertexSBAPointXYZ() MapPoint的mWorldPos
 * 2.edge:   g2o::EdgeSE3ProjectXYZ(),BaseBinaryEdge
 *           ----Vertex:待优化CurrentKF的Tcw
 *           ----Vertex:待优化MapPoint的mWorldPos
 *           ----Measurement:MapPoint在CurrentKF中的二维坐标(u,v)
 *           ----InfoMatrix:invSigma2 与特征点所在尺度有关
 *  
 * @param vpKFs 存放所有KeyFrames的容器
 * @param vpMP 存放所有MapPoints的容器
 * @param nIterations 迭代的次数
 * @param pbStopFlag 是否停止优化的标志位
 * @param nLoopKF 在id为nLoopKF处进行BA
 * @param bRobust 是否使用核函数
 */
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size()); //不参与优化的点

    //step1: 初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver; //位姿6维，路标点为3维

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); //LM算法进行优化
    optimizer.setAlgorithm(solver);

    if (pbStopFlag) //是否强制停止
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0; //记录添加到优化器的顶点的最大关键帧ID

    // step2:Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++) //遍历提供的所有关键帧，添加keyframe pose作为顶点误差变量
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad()) //去除无效的
            continue;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap(); //pose SE3
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose())); //设置pose顶点误差变量的初值
        vSE3->setId(pKF->mnId);                                  //顶点的ID
        vSE3->setFixed(pKF->mnId == 0);                          //如果是第0帧，就不优化这个顶点误差变量
        optimizer.addVertex(vSE3);                               //将配置好的顶点添加到optimizer
        if (pKF->mnId > maxKFid)                                 //更新maxKFid
            maxKFid = pKF->mnId;
    }

    //卡方分布 95%以上可信度时的阈值
    const float thHuber2D = sqrt(5.99);  //2自由度
    const float thHuber3D = sqrt(7.815); //3自由度

    // step3:Set MapPoint vertices
    //一边添加MapPoint的顶点，一边添加边
    for (size_t i = 0; i < vpMP.size(); i++) //遍历提供的所有MapPoint，添加MapPoint position 作为顶点误差变量
    {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();  //point xyz
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos())); //顶点的初值,将cv::Mat格式数据转换为Eigen::Vector3d类型
        const int id = pMP->mnId + maxKFid + 1;                         //顶点的id，与pose的区别开
        vPoint->setId(id);
        vPoint->setMarginalized(true); //进行schur消元，是否利用稀疏化加速
        optimizer.addVertex(vPoint);   //将配置好的顶点添加到optimizer

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //step4: Set Edges 在遍历地图点时，将该点与可观测到他的KF进行连接
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) //遍历此MapPoint能被看到的所有KeyFrame，向优化器添加误差边
        {

            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0) //单目或RGBD相机
            {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ(); //SE3->XYZ

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));        //添加和这条边相连接的MapPoint顶点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId))); //添加和这条边相连接的KeyFrame顶点
                e->setMeasurement(obs);                                                                      //添加观测值

                //根据MapPoint所在的高斯金字塔尺度设置信息矩阵
                //信息矩阵表明了这个约束的观测在各个维度(u,v)上的可信度，系统中两个坐标的可信度是相同的
                //可信度与特征点在图像金字塔中的层数有关，图层越高，可信度越差
                //使用平方是为了避免信息矩阵中元素为负
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) //如果使用核函数
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //重投影误差为2自由度，因此使用2自由度的卡方分布值，如果误差相差大于1个像素，认为该点不可靠
                    //核函数为了避免误差点出现数值过大的增长
                    rk->setDelta(thHuber2D);
                }

                //向边添加内参
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e); //添加边
            }
            else //双目
            {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ(); //双目相机的输入边

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0) //如果该MapPoint没有任何一个KeyFrame观测到，则将其移除
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }

    //step5:Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    //step6:Recover optimized data
    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId)); //确保数据类型正确
        g2o::SE3Quat SE3quat = vSE3->estimate();                                                       //取出优化变量
        if (nLoopKF == 0)                                                                              //如果整个GlobalBA是在初始帧调用的话，KF位姿可以直接写入
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else //闭环中调用，将优化后的位姿写入变量mTcwGBA中，临时保存
        {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++) //更新MapPoint
    {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint *pMP = vpMP[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (nLoopKF == 0) //如果整个GlobalBA是在初创地图时调用的话，地图点位置可以直接写入
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else //如果是闭环过程中调用，将优化后的地图点写入变量mPosGBA，临时保存一下
        {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

/**
 * @brief 只优化Frame的Tcw，不优化MapPoints
 *        更新pFrame->mvbOutlier
 *        更新了pFrame的位姿
 * 
 * @param pFrame 
 * @return int inliers的数量
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    //step1:构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver; //位姿六维，路标点三维

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    //step2:Set Frame vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw)); //待优化当前帧的Tcw
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    //卡方分布
    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    //step3：添加边
    {
        //地图点加锁，需要使用地图点构造顶点和边，因此不希望构造过程中地图点被改写
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)//遍历所有地图点
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP)//如果地图点未被剔除
            {
                // Monocular observation
                if (pFrame->mvuRight[i] < 0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;//对地图点的观测
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    //该节点只优化位姿Pose
                    g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];//可信度与图层相关
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();//地图点位置为迭代的初始值
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double, 3, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
        }
    }

    if (nInitialCorrespondences < 3) //如果只添加了3条边
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10}; //每次迭代的次数

    int nBad = 0;//bad地图点个数
    for (size_t it = 0; it < 4; it++)//四次优化
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0); //对level为0的边进行优化
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)//遍历参与优化的每一条误差边
        {
            g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx])//如果这条误差边为outlier
            {
                e->computeError(); //值计算active edge的误差
            }

            const float chi2 = e->chi2();//error*Omega*error 这个点的误差大小

            if (chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1); //设置为outlier
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0); //设置为inlier
            }

            if (it == 2)
                e->setRobustKernel(0); //只有前两次优化使用核函数，因为重投影误差已经有明显下降了
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // step5:Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;//返回内点个数
}

/**
 * @brief 主要用于LocalMapping线程的局部BA优化 
 *        1.将Covisibility graph中与pKF连接的关键帧放入lLocalKeyFrames作为g2o图的顶点
 *        2.将被lLocalKeyFrames看到的MapPoint放入lLocalKeyFrames中作为g2o图的顶点
 *        3.lFixedCameras储存着能看到lLocalMapPoints，但是又不在lLocalKeyFrames里的关键帧，作为g2o图的顶点
 *        4.将lLocalMapPoints里的MapPoint的每个观测作为g2o图的边
 * @param pKF 
 * @param pbStopFlag 
 * @param pMap 
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
{
    // step1:Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame *> lLocalKeyFrames; //将covisibility graph中与pKF连接的关键帧放入lLocalKeyFrames

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // step2:Local MapPoints seen in Local KeyFrames
    list<MapPoint *> lLocalMapPoints;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) //将被lLocalKeyFrames看到的MapPoint放入lLocalKeyFrames中
    {
        vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId; //将参与LocalBA的每一个关键帧的mnBALocalForKF设置为当前帧的Id，防止重复添加地图点
                    }
        }
    }

    // step3: Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame *> lFixedCameras; //lFixedCameras储存着能看到lLocalMapPoints，但是又不在lLocalKeyFrames里的关键帧，二级共视帧不被优化
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)//遍历地图点
    {
        map<KeyFrame *, size_t> observations = (*lit)->GetObservations();//观测到该MapPoint的KF和MapPoint在KF中的索引
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)//遍历观测到该地图点的关键帧
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)//不属于局部关键帧且且还未固定的关键帧
            {
                pKFi->mnBAFixedForKF = pKF->mnId; //能观测到局部地图点但是不属于局部BA范围的关键帧mnBAFixedForKF（触发局部BA的当前关键帧）的mnID
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // step4:Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;//记录参与局部BA最大关键帧mnID,为了优化时地图点与关键帧进行区分

    // step5:Set Local KeyFrame vertices
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) //将lLocalKeyFrames关键帧的位姿设置为g2o图的顶点

    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));//初始优化位姿
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0);//第一帧位姿不参与优化
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // step6:Set Fixed KeyFrame vertices
    //加入不优化的位姿顶点，增加约束项
    for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) //将lFixedCameras里的关键帧的位姿设置为g2o图的顶点
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true); //这些关键帧在局部BA中不优化
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // step7:Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) //将lLocalMapPoints里的mappoint空间位置作为g2o图的顶点

    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);//将三维点边缘化
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();//地图点与KF的索引

        //step8: edges
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if (pKFi->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;
    //step9: 开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);//迭代5次

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {

        // Check inlier observations
        // step10:检测outlier，并设置下次不优化
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)//遍历所有单目误差边
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive()) //基于卡方检测计算出的阈值
            {
                e->setLevel(1); //不优化
            }

            e->setRobustKernel(0); //不使用核函数
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        //step11:排除误差较大的outlier再优化
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // step12:Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) //在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty()) //剔除
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);//在关键帧中删除对地图点的观测
            pMPi->EraseObservation(pKFi);//在地图点中删除关键帧的观测
        }
    }

    // Recover optimized data
    // step13:优化后更新关键帧pose以及MapPoints的位置、平均观测方向等等属性

    //Keyframes
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKF = *lit;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

/**
 * @brief 定点为map中所有KeyFrame
 *        边为LoopConnections中的连接关系，以及essential graph中的边：1.扩展树 2.闭环连接关系 3.共视关系好的连接关系（共视点为100）
 * 
 * @param pMap 全局地图
 * @param pLoopKF 闭环匹配上的关键帧
 * @param pCurKF 当前关键帧
 * @param NonCorrectedSim3 未经过sim3传播调整过的关键帧pose
 * @param CorrectedSim3 经过sim3传播调整过的关键帧pose
 * @param LoopConnections 因闭环时MapPoints调整而新生成的边
 * @param bFixScale 
 */
void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *>> &LoopConnections, const bool &bFixScale)
{
    // step1:Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);//第一次迭代的初始lambda值，如果未指定会自动计算一个合适的值
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();//用于添加顶点

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);          //经过Sim3传播调整，未经过优化的KeyFrame的pose
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1); //经过Sim3传播调整，经过优化的KeyFrame的pose
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;//两个KF之间共视关系的权重（共视点数目的最小值）

    // step2:Set KeyFrame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) //将map中的所有关键帧添加为g2o的顶点
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;//用来设置顶点ID

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF); 

        if (it != CorrectedSim3.end())//尽可能使用在闭环时通过sim3调整过的位姿
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else//如果该关键帧在闭环时没用通过sim3调整过，就用跟踪时的位姿
        {
            Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw, tcw, 1.0);//尺度为1
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if (pKF == pLoopKF) //闭环匹配上的帧不进行位姿优化（认为是准的，作为基准）
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges; //闭环优化出现的新的关键帧和关键帧之间的连接关系，firstid数较小的顶点

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    // step3:Set Loop edges 闭环时因为地图点调整而出现的新关键帧连接关系
    // 第一种：遍历每一个闭环生成的边i
    for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame *> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        //遍历每一个新添加的关键帧连接关系j
        for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)//1.i不是当前帧或者j不是闭环帧 2.共视程度小，不足以构成约束的边
                continue;
            //ij其耗时当前帧以及闭环帧 或者ij共视程度大
            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi; //i与j之间的位姿

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji); //根据两个pose顶点的位姿算出相对位姿作为边???

            e->information() = matLambda;//信息矩阵为单位阵

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));//id小的在前，id大的在后
        }
    }

    // step4:Set normal edges 添加跟踪时形成的边、闭环匹配成功形成的边
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) //遍历vpKFs
    {
        KeyFrame *pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if (iti != NonCorrectedSim3.end())//如果在未修正的sim3中
            Swi = (iti->second).inverse();
        else//使用跟踪时的位姿
            Swi = vScw[nIDi].inverse();

        KeyFrame *pParentKF = pKF->GetParent();//和当前帧共视程度最高的关键帧

        // Spanning tree edge
        // 第二种：添加扩展树的边
        if (pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if (itj != NonCorrectedSim3.end())//优先使用未经过sim3传播调整的位姿
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3(); //将vpKFs和其在spanningtree中的父节点在g2o图中连接起来形成一条误差边
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 第三种：添加当前帧与闭环匹配帧之间的连接关系
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges(); // 添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边（当前帧与闭环匹配帧之间的连接关系）

        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)//避免重复添加
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3 *el = new g2o::EdgeSim3(); //将vpKFs和其形成闭环的帧在g2o图中连接起来形成一条误差边
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        // 第四种：添加共视程度高的关键帧进行优化
        const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat); // 最有很好共视关系的关键帧也作为边进行优化
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            //避免重复添加：最小生成树中的父子关键帧关系以及和当前遍历到的关键帧构成了回环关系
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))//判断是否已经添加
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    //更新地图要上锁
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 遍历地图中的所有关键帧
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        pKFi->SetPose(Tiw);//将优化好的位姿写入关键帧
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 根据优化得到的关键帧位姿，地图点根据参考帧前后关系调整自己的位置
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;
        if (pMP->mnCorrectedByKF == pCurKF->mnId)//如果被调整过，那么使用调整它的KFid
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        g2o::Sim3 Srw = vScw[nIDr];                   // 得到MapPoint参考关键帧优化前的位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr]; // 得到MapPoint参考关键帧优化后的位姿

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

/**
 * @brief 形成闭环时固定地图点，只优化Sim3
 * 
 * @param pKF1 当前帧
 * @param pKF2 闭环候选帧
 * @param vpMatches1 pKF1的特征点和pKF2的MapPoint匹配情况
 * @param g2oS12 两个关键帧间的Sim3变换
 * @param th2 卡方检验是否为误差边的阈值
 * @param bFixScale 是否进行尺度优化
 * @return int 优化之后匹配点中内点的个数
 */
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    //构造线性方程求解器，Hx=-b的求解器
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    //dense的求解器
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap(); //添加sim3位姿作为顶点
    vSim3->_fix_scale = bFixScale; //是否固定尺度
    vSim3->setEstimate(g2oS12); //添加Sim3位姿顶点误差变量
    vSim3->setId(0);
    vSim3->setFixed(false);//需要优化sim3位姿
    vSim3->_principle_point1[0] = K1.at<float>(0, 2); //将内参导入顶点
    vSim3->_principle_point1[1] = K1.at<float>(1, 2);
    vSim3->_focal_length1[0] = K1.at<float>(0, 0);
    vSim3->_focal_length1[1] = K1.at<float>(1, 1);
    vSim3->_principle_point2[0] = K2.at<float>(0, 2);
    vSim3->_principle_point2[1] = K2.at<float>(1, 2);
    vSim3->_focal_length2[0] = K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = K2.at<float>(1, 1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches(); //获得pKF1的所有MapPoint
    vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;//pKF2对应的地图点到pKF1的投影
    vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;//pKF1对应的地图点到pKF2的投影
    vector<size_t> vnIndexEdge;//边索引

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);

    const float deltaHuber = sqrt(th2);//核函数

    int nCorrespondences = 0;

    for (int i = 0; i < N; i++) //将匹配转化为归一化3d点作为g2o的顶点
    {
        if (!vpMatches1[i])
            continue;

        MapPoint *pMP1 = vpMapPoints1[i];
        MapPoint *pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2); //找到匹配点在pKF2观测到的哪个点

        if (pMP1 && pMP2)
        {
            if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)//如果这对匹配点都是好点，并且对应的2d特征点存在
            {
                g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);//地图点不优化
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2 正向投影
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));//待优化sim3
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1 逆向投影
        Eigen::Matrix<double, 2, 1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ(); //inverse of sim3

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    // 卡方检验剔除误差大的边
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)//正向反向投影任意超过阈值就删除该边
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
            vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
            nBad++;//累计删除的边数目
        }
    }

    int nMoreIterations;
    //如果有误差边被删除，则还需要多迭代几次
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;
    //如果经过剔除后匹配关系过少，则放弃优化
    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization(); //剔除后再次优化
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
        }
        else
            nIn++;//第二次优化后的内点个数
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}

} // namespace ORB_SLAM2
