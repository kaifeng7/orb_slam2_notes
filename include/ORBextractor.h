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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace ORB_SLAM2
{

/**
 * @brief 维护四叉树结构，用于存放节点数据
 * 
 */
class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;//记录属于该节点的所有特征点
    cv::Point2i UL, UR, BL, BR; //up left,up right,bottom left,bottom right，四个顶点坐标
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore; //whether not to continue dividing
};

/**
 * @brief 
 * 
 */
class ORBextractor
{
public:
    enum
    {
        HARRIS_SCORE = 0,
        FAST_SCORE = 1
    };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    ~ORBextractor() {}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors);//进行关键点的提取和描述子的计算

    int inline GetLevels()
    {
        return nlevels;
    }

    float inline GetScaleFactor()
    {
        return scaleFactor;
    }

    std::vector<float> inline GetScaleFactors()
    {
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors()
    {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares()
    {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares()
    {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
    std::vector<cv::Point> pattern;//图像块

    int nfeatures;//期望提取的关键点数量
    double scaleFactor;//相邻两层金字塔之间的相对尺度因子，金字塔越往上的图像每个像素代表的范围越大
    int nlevels;//金字塔层数
    int iniThFAST;//提取fast特征点的默认阈值
    int minThFAST;//如果使用iniThFAST提取不到特征点，则使用最小阈值再次提取

    std::vector<int> mnFeaturesPerLevel;//每一层关键点的数量

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;//每一层相对于第一层的尺度因子
    std::vector<float> mvInvScaleFactor;//尺度因子的逆
    std::vector<float> mvLevelSigma2;//尺度因子的平方
    std::vector<float> mvInvLevelSigma2;//尺度因子平方的逆
};

} // namespace ORB_SLAM2

#endif
