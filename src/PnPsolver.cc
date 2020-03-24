/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#include <iostream>

#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{

/**
 * @brief Construct a new Pn Psolver:: Pn Psolver object
 * 
 * @param F current frame
 * @param vpMapPointMatches 匹配点 
 */
PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches) : pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
																					mnIterations(0), mnBestInliers(0), N(0)
{
	//初始化各个容器
	mvpMapPointMatches = vpMapPointMatches;
	mvP2D.reserve(F.mvpMapPoints.size());
	mvSigma2.reserve(F.mvpMapPoints.size());
	mvP3Dw.reserve(F.mvpMapPoints.size());
	mvKeyPointIndices.reserve(F.mvpMapPoints.size());
	mvAllIndices.reserve(F.mvpMapPoints.size());

	int idx = 0;

	for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
	{
		MapPoint *pMP = vpMapPointMatches[i]; //依次获得一个MapPoint

		if (pMP)
		{
			if (!pMP->isBad())
			{
				const cv::KeyPoint &kp = F.mvKeysUn[i]; //得到2d points

				mvP2D.push_back(kp.pt);
				mvSigma2.push_back(F.mvLevelSigma2[kp.octave]); //记录特征点是在哪一层提取出来的

				cv::Mat Pos = pMP->GetWorldPos();
				mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));

				mvKeyPointIndices.push_back(i); //记录好的特征点在原始特征点容器中的索引（跳过坏点，空点）
				mvAllIndices.push_back(idx);	//记录好的特征点索引（连续[0,1,2,3...]）

				idx++;
			}
		}
	}

	// Set camera calibration parameters
	fu = F.fx;
	fv = F.fy;
	uc = F.cx;
	vc = F.cy;

	SetRansacParameters();
}

PnPsolver::~PnPsolver()
{
	delete[] pws;
	delete[] us;
	delete[] alphas;
	delete[] pcs;
}

/**
 * @brief set Ransac parameters
 * 
 * @param probability 
 * @param minInliers 
 * @param maxIterations 
 * @param minSet 
 * @param epsilon 
 * @param th2 
 */
void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
	mRansacProb = probability;
	mRansacMinInliers = minInliers;
	mRansacMaxIts = maxIterations;
	mRansacEpsilon = epsilon;
	mRansacMinSet = minSet;

	N = mvP2D.size(); // number of correspondences

	mvbInliersi.resize(N);

	// Adjust Parameters according to number of correspondences
	int nMinInliers = N * mRansacEpsilon;
	if (nMinInliers < mRansacMinInliers)
		nMinInliers = mRansacMinInliers;
	if (nMinInliers < minSet)
		nMinInliers = minSet;
	mRansacMinInliers = nMinInliers; //避免最少内阈值设置的过小

	if (mRansacEpsilon < (float)mRansacMinInliers / N)
		mRansacEpsilon = (float)mRansacMinInliers / N;

	// Set RANSAC iterations according to probability, epsilon, and max iterations
	int nIterations;

	if (mRansacMinInliers == N)
		nIterations = 1;
	else
		nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 3)));

	mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

	mvMaxError.resize(mvSigma2.size()); //每个2d点对应不同的误差阈值
	for (size_t i = 0; i < mvSigma2.size(); i++)
		mvMaxError[i] = mvSigma2[i] * th2; //不同的尺度设置不同的最大偏差
}

cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
{
	bool bFlag;
	return iterate(mRansacMaxIts, bFlag, vbInliers, nInliers);
}

/**
 * @brief 内部会有ransac迭代，外部会以while的方式多次调用iterate
 * 
 * @param nIterations 
 * @param bNoMore 
 * @param vbInliers 
 * @param nInliers 
 * @return cv::Mat 
 */
cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
	bNoMore = false;
	vbInliers.clear();
	nInliers = 0;

	set_maximum_number_of_correspondences(mRansacMinSet);

	if (N < mRansacMinInliers) //如果当前所有点小于阈值
	{
		bNoMore = true;
		return cv::Mat();
	}

	vector<size_t> vAvailableIndices;

	int nCurrentIterations = 0; //当前迭代会调用RANSAC的次数
	while (mnIterations < mRansacMaxIts || nCurrentIterations < nIterations)
	{
		nCurrentIterations++;
		mnIterations++;
		reset_correspondences();

		vAvailableIndices = mvAllIndices; //每次迭代将所有的特征匹配复制一份

		// Get min set of points
		for (short i = 0; i < mRansacMinSet; ++i)
		{
			int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

			int idx = vAvailableIndices[randi];

			add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y); //将对应的3D-2D压入pws和us

			vAvailableIndices[randi] = vAvailableIndices.back(); //避免抽取同一个数据参与ransac
			vAvailableIndices.pop_back();
		}

		// Compute camera pose
		compute_pose(mRi, mti);

		// Check inliers
		CheckInliers();

		if (mnInliersi >= mRansacMinInliers) //如果内点数大于阈值
		{
			// If it is the best solution so far, save it
			if (mnInliersi > mnBestInliers) //记录inlier个数最多的一个解
			{
				mvbBestInliers = mvbInliersi;
				mnBestInliers = mnInliersi;

				cv::Mat Rcw(3, 3, CV_64F, mRi);
				cv::Mat tcw(3, 1, CV_64F, mti);
				Rcw.convertTo(Rcw, CV_32F);
				tcw.convertTo(tcw, CV_32F);
				mBestTcw = cv::Mat::eye(4, 4, CV_32F);
				Rcw.copyTo(mBestTcw.rowRange(0, 3).colRange(0, 3));
				tcw.copyTo(mBestTcw.rowRange(0, 3).col(3));
			}

			if (Refine()) //将所有符合inlier的3D-2D匹配点一起计算PnP求解Rt
			{
				nInliers = mnRefinedInliers;
				vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
				for (int i = 0; i < N; i++)
				{
					if (mvbRefinedInliers[i])
						vbInliers[mvKeyPointIndices[i]] = true;
				}
				return mRefinedTcw.clone();
			}
		}
	}

	if (mnIterations >= mRansacMaxIts)
	{
		bNoMore = true;
		if (mnBestInliers >= mRansacMinInliers)
		{
			nInliers = mnBestInliers;
			vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
			for (int i = 0; i < N; i++)
			{
				if (mvbBestInliers[i])
					vbInliers[mvKeyPointIndices[i]] = true;
			}
			return mBestTcw.clone();
		}
	}

	return cv::Mat();
}

bool PnPsolver::Refine()
{
	vector<int> vIndices;
	vIndices.reserve(mvbBestInliers.size());

	for (size_t i = 0; i < mvbBestInliers.size(); i++)
	{
		if (mvbBestInliers[i])
		{
			vIndices.push_back(i);
		}
	}

	set_maximum_number_of_correspondences(vIndices.size());

	reset_correspondences();

	for (size_t i = 0; i < vIndices.size(); i++)
	{
		int idx = vIndices[i];
		add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y);
	}

	// Compute camera pose
	compute_pose(mRi, mti);

	// Check inliers
	CheckInliers();

	mnRefinedInliers = mnInliersi;
	mvbRefinedInliers = mvbInliersi;

	if (mnInliersi > mRansacMinInliers)
	{
		cv::Mat Rcw(3, 3, CV_64F, mRi);
		cv::Mat tcw(3, 1, CV_64F, mti);
		Rcw.convertTo(Rcw, CV_32F);
		tcw.convertTo(tcw, CV_32F);
		mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
		Rcw.copyTo(mRefinedTcw.rowRange(0, 3).colRange(0, 3));
		tcw.copyTo(mRefinedTcw.rowRange(0, 3).col(3));
		return true;
	}

	return false;
}

/**
 * @brief 统计和记录inlier个数以及符合inlier的点
 * 
 */
void PnPsolver::CheckInliers()
{
	mnInliersi = 0;

	for (int i = 0; i < N; i++)
	{
		cv::Point3f P3Dw = mvP3Dw[i];
		cv::Point2f P2D = mvP2D[i];

		//world->camera
		float Xc = mRi[0][0] * P3Dw.x + mRi[0][1] * P3Dw.y + mRi[0][2] * P3Dw.z + mti[0];
		float Yc = mRi[1][0] * P3Dw.x + mRi[1][1] * P3Dw.y + mRi[1][2] * P3Dw.z + mti[1];
		float invZc = 1 / (mRi[2][0] * P3Dw.x + mRi[2][1] * P3Dw.y + mRi[2][2] * P3Dw.z + mti[2]);

		//camera->image
		double ue = uc + fu * Xc * invZc;
		double ve = vc + fv * Yc * invZc;

		//参差大小
		float distX = P2D.x - ue;
		float distY = P2D.y - ve;

		float error2 = distX * distX + distY * distY;

		if (error2 < mvMaxError[i])
		{
			mvbInliersi[i] = true;
			mnInliersi++;
		}
		else
		{
			mvbInliersi[i] = false;
		}
	}
}

/**
 * @brief 创建用于RANSAC的内存空间，默认为4组3D-2D的对应点
 * 
 * @param n 
 */
void PnPsolver::set_maximum_number_of_correspondences(int n)
{
	if (maximum_number_of_correspondences < n) //如果当前设置值过小，则重新设置，重新初始化pws，us，alphas，pcs的大小
	{
		if (pws != 0)
			delete[] pws;
		if (us != 0)
			delete[] us;
		if (alphas != 0)
			delete[] alphas;
		if (pcs != 0)
			delete[] pcs;

		maximum_number_of_correspondences = n;
		pws = new double[3 * maximum_number_of_correspondences];
		us = new double[2 * maximum_number_of_correspondences];
		alphas = new double[4 * maximum_number_of_correspondences];
		pcs = new double[3 * maximum_number_of_correspondences];
	}
}

/**
 * @brief 重置空间大小
 * 
 */
void PnPsolver::reset_correspondences(void)
{
	number_of_correspondences = 0;
}

/**
 * @brief 加入3d-2d组
 * 
 * @param X 
 * @param Y 
 * @param Z 
 * @param u 
 * @param v 
 */
void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
	pws[3 * number_of_correspondences] = X;
	pws[3 * number_of_correspondences + 1] = Y;
	pws[3 * number_of_correspondences + 2] = Z;

	us[2 * number_of_correspondences] = u;
	us[2 * number_of_correspondences + 1] = v;

	number_of_correspondences++;
}

/**
 * @brief 获得4个控制点坐标，存在4*3的二维数组cws中
 * 
 */
void PnPsolver::choose_control_points(void)
{
	// Take C0 as the reference points centroid:
	// 1.C0为模型坐标系的中心点
	cws[0][0] = cws[0][1] = cws[0][2] = 0;
	for (int i = 0; i < number_of_correspondences; i++)
		for (int j = 0; j < 3; j++)
			cws[0][j] += pws[3 * i + j];

	for (int j = 0; j < 3; j++)
		cws[0][j] /= number_of_correspondences;

	// Take C1, C2, and C3 from PCA on the reference points:
	// 2.c1，c2，c3通过PCA分解得到
	CvMat *PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);//每个点与中心点的残差值

	double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
	CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
	CvMat DC = cvMat(3, 1, CV_64F, dc);
	CvMat UCt = cvMat(3, 3, CV_64F, uct);

	//将存在pws中的3D点减去第一个控制点的坐标（相当于把第一个控制点作为原点），存入pw0
	for (int i = 0; i < number_of_correspondences; i++)
		for (int j = 0; j < 3; j++)
			PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

	//svd分解p`p可以得到P的主成分
	//齐次线性最小二乘求解
	cvMulTransposed(PW0, &PW0tPW0, 1);
	cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

	cvReleaseMat(&PW0);

	//得到c1，c2，c3三个3D控制点，最后加上之前减掉的第一个控制点的偏移量
	for (int i = 1; i < 4; i++)
	{
		double k = sqrt(dc[i - 1] / number_of_correspondences);
		for (int j = 0; j < 3; j++)
			cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
	}
}

/**
 * @brief 求解四个控制点的系数alphas
 * 		  (a2 a3 a4)' = inverse(cws2-cws1 cws3-cws1 cws4-cws1)*(pws-cws1) a1 = 1-a2-a3-a4
 * 		  每一个3d点，都有一组alphas与之对应
 * 		  cws1 cws2 cws3 cws4 为四个控制点的坐标
 */
void PnPsolver::compute_barycentric_coordinates(void)
{
	double cc[3 * 3], cc_inv[3 * 3];
	CvMat CC = cvMat(3, 3, CV_64F, cc);//[c1-c0,c2-c0,c3-c0]
	CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

	for (int i = 0; i < 3; i++)
		for (int j = 1; j < 4; j++)
			cc[3 * i + j - 1] = cws[j][i] - cws[0][i];

	cvInvert(&CC, &CC_inv, CV_SVD);
	double *ci = cc_inv;
	for (int i = 0; i < number_of_correspondences; i++)
	{
		double *pi = pws + 3 * i;
		double *a = alphas + 4 * i;

		//[a1,a2,a3]^T = [c1' c2' c3']^-1 * [x-x']
		for (int j = 0; j < 3; j++)
			a[1 + j] =
				ci[3 * j] * (pi[0] - cws[0][0]) +
				ci[3 * j + 1] * (pi[1] - cws[0][1]) +
				ci[3 * j + 2] * (pi[2] - cws[0][2]);
		a[0] = 1.0f - a[1] - a[2] - a[3];
	}
}

/**
 * @brief 构成M矩阵
 * 
 * 		  对每个一3d点 ，i=[0.1.2.4]，转换成对应质点模型，并转化到像素坐标系下
 * 		  [ai0*fx,0,ai0*(cx-ui)],[ai1*fx,0,ai1*(cx-ui)],[ai2*fx,0,ai2*(cx-ui)],[ai3*fx,0,ai3*(cx-ui)]
 * 		  [0,ai0*fy,ai0*(cy-vi)],[0,ai1*fy,ai1*(cy-vi)],[0,ai2*fy,ai2*(cy-vi)],[0,ai3*fy,ai3*(cy-vi)]
 * @param M 
 * @param row 多少个3d点
 * @param as a0,a1,a2,a3
 * @param u 
 * @param v 
 */
void PnPsolver::fill_M(CvMat *M,const int row, const double *as, const double u, const double v)
{
	double *M1 = M->data.db + row * 12;//指向M的前12个元素，即row=0
	double *M2 = M1 + 12;//指向M的后12个元素，即row=1

	for (int i = 0; i < 4; i++)
	{
		M1[3 * i] = as[i] * fu;
		M1[3 * i + 1] = 0.0;
		M1[3 * i + 2] = as[i] * (uc - u);

		M2[3 * i] = 0.0;
		M2[3 * i + 1] = as[i] * fv;
		M2[3 * i + 2] = as[i] * (vc - v);
	}
}

/**
 * @brief 每个控制点在相机坐标系下都表示为特征向量乘以beta的形式
 * 
 * @param betas 
 * @param ut 
 */
void PnPsolver::compute_ccs(const double *betas, const double *ut)
{
	for (int i = 0; i < 4; i++)
		ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

	for (int i = 0; i < 4; i++)
	{
		const double *v = ut + 12 * (11 - i);
		for (int j = 0; j < 4; j++)
			for (int k = 0; k < 3; k++)
				ccs[j][k] += betas[i] * v[3 * j + k];
	}
}

/**
 * @brief 根据控制点和camera下每个点与控制点之间的关系，恢复出所有3d点在相机坐标系下的坐标
 * 
 */
void PnPsolver::compute_pcs(void)
{
	for (int i = 0; i < number_of_correspondences; i++)
	{
		double *a = alphas + 4 * i;
		double *pc = pcs + 3 * i;

		for (int j = 0; j < 3; j++)
			pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
	}
}

/**
 * @brief EPnP算法求解Rt
 * 
 * @param R 
 * @param t 
 * @return double 
 */
double PnPsolver::compute_pose(double R[3][3], double t[3])
{
	//1.获得EPNP算法中的四个控制点（构成模型坐标系）
	choose_control_points();
	//2.计算世界坐标系下的每个3d点用4个控制点线性表达时的系数alphas
	compute_barycentric_coordinates();

	//3.构造M矩阵，即3d点转换为质心坐标系下，并转换到像素坐标系
	CvMat *M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);//2*n维

	for (int i = 0; i < number_of_correspondences; i++)
		fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

	double mtm[12 * 12], d[12], ut[12 * 12];
	CvMat MtM = cvMat(12, 12, CV_64F, mtm);
	CvMat D = cvMat(12, 1, CV_64F, d);//特征值
	CvMat Ut = cvMat(12, 12, CV_64F, ut);//特征向量

	//4.求解Mx=0
	cvMulTransposed(M, &MtM, 1);
	//svd分解M`M，最小二乘法得到相机坐标系下四个不带尺度的控制点ut
	//ut的每一行对应一组可能的解
	//最小特征值对应的特征向量最接近待求的解，由于噪声和约束不足的问题，导致真正的解可能是多个特征向量的线性叠加
	//svd分解按照特征值大小降序排列，越往下的排列的特征向量越优
	//只考虑N=4的情况
	cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

	cvReleaseMat(&M);

	//默认N=4
	//通过最小二乘法（L*betas=Rho）来求解尺度betas
	//L6x10 * Betas10x1 = Rho_6x1
	double l_6x10[6 * 10], rho[6];
	CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);//camera坐标系下的四点距离
	CvMat Rho = cvMat(6, 1, CV_64F, rho);//world坐标系下的四点距离

	//Betas 10x1 = [b00 b01 b11 b02 b12 b22 b03 b13 b23 b33]
	//L 6x10= [dv00,2*dv01,dv11,2*dv02,2*dv12,dv22,2*dv03,2*dv13,2*dv23,dv33]
	//4个控制点之间共有6个距离，因此为6*10
	compute_L_6x10(ut, l_6x10);
	compute_rho(rho);

	double Betas[4][4], rep_errors[4];
	double Rs[4][3][3], ts[4][3];

	//通过最小二乘法纠结部分betas(全求出来会有冲突)
	//通过优化得到剩下的betas
	
	//近似解一
	find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
	gauss_newton(&L_6x10, &Rho, Betas[1]);
	rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

	//近似解二
	find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
	gauss_newton(&L_6x10, &Rho, Betas[2]);
	rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

	//近似解三
	find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
	gauss_newton(&L_6x10, &Rho, Betas[3]);
	rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

	int N = 1;
	if (rep_errors[2] < rep_errors[1])
		N = 2;
	if (rep_errors[3] < rep_errors[N])
		N = 3;

	copy_R_and_t(Rs[N], ts[N], R, t);

	return rep_errors[N];
}

void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
							 double R_dst[3][3], double t_dst[3])
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			R_dst[i][j] = R_src[i][j];
		t_dst[i] = t_src[i];
	}
}

double PnPsolver::dist2(const double *p1, const double *p2)
{
	return (p1[0] - p2[0]) * (p1[0] - p2[0]) +
		   (p1[1] - p2[1]) * (p1[1] - p2[1]) +
		   (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double *v1, const double *v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/**
 * @brief 重投影误差
 * 
 * @param R 旋转
 * @param t 平移
 * @return double 
 */
double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
	double sum2 = 0.0;

	for (int i = 0; i < number_of_correspondences; i++)
	{
		double *pw = pws + 3 * i;
		double Xc = dot(R[0], pw) + t[0];
		double Yc = dot(R[1], pw) + t[1];
		double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
		double ue = uc + fu * Xc * inv_Zc;
		double ve = vc + fv * Yc * inv_Zc;
		double u = us[2 * i], v = us[2 * i + 1];

		sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
	}

	return sum2 / number_of_correspondences;
}

/**
 * @brief 根据世界坐标系下的四个控制点与camera下对应的四个控制点，求Rt
 * 		  [U,s,Vt] = svd(A*B') A为pci列向量构成的矩阵，B为pci行向量构成的矩阵
 * 		  R = U*Vt
 * 		  t = pc0 - R*pw0 pc0和pw0分别为camera和world下3d点的中心坐标 
 * 
 * @param R 
 * @param t 
 */
void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
	double pc0[3], pw0[3];//两个中心点

	pc0[0] = pc0[1] = pc0[2] = 0.0;
	pw0[0] = pw0[1] = pw0[2] = 0.0;

	for (int i = 0; i < number_of_correspondences; i++)
	{
		const double *pc = pcs + 3 * i;
		const double *pw = pws + 3 * i;

		for (int j = 0; j < 3; j++)
		{
			pc0[j] += pc[j];
			pw0[j] += pw[j];
		}
	}
	for (int j = 0; j < 3; j++)
	{
		pc0[j] /= number_of_correspondences;
		pw0[j] /= number_of_correspondences;
	}

	//     |pc_x_0, pc_x_1, pc_x_2,....pc_x_n|
    // A = |pc_y_0, pc_y_1, pc_y_2,....pc_x_n|
    //     |pc_z_0, pc_z_1, pc_z_2,....pc_x_n|

    //     |pw_x_0, pw_x_1, pw_x_2,....pw_x_n|
    // B = |pw_y_0, pw_y_1, pw_y_2,....pw_x_n|
    //     |pw_z_0, pw_z_1, pw_z_2,....pw_x_n|
	
	double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
	CvMat ABt = cvMat(3, 3, CV_64F, abt);
	CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
	CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
	CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

	//计算A * B^t
	cvSetZero(&ABt);
	for (int i = 0; i < number_of_correspondences; i++)
	{
		double *pc = pcs + 3 * i;
		double *pw = pws + 3 * i;

		for (int j = 0; j < 3; j++)
		{
			abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
			abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
			abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
		}
	}

	cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

	const double det =
		R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
		R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

	if (det < 0)
	{
		R[2][0] = -R[2][0];
		R[2][1] = -R[2][1];
		R[2][2] = -R[2][2];
	}

	t[0] = pc0[0] - dot(R[0], pw0);
	t[1] = pc0[1] - dot(R[1], pw0);
	t[2] = pc0[2] - dot(R[2], pw0);
}

void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
	cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
	cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
	cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

/**
 * @brief 随机取一个camera下的3d点，如果z<0,则表面3d点都在相机后面，则3d点整体取负号
 * 
 */
void PnPsolver::solve_for_sign(void)
{
	if (pcs[2] < 0.0)
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 3; j++)
				ccs[i][j] = -ccs[i][j];

		for (int i = 0; i < number_of_correspondences; i++)
		{
			pcs[3 * i] = -pcs[3 * i];
			pcs[3 * i + 1] = -pcs[3 * i + 1];
			pcs[3 * i + 2] = -pcs[3 * i + 2];
		}
	}
}

/**
 * @brief 求解R和t
 * 
 * @param ut 
 * @param betas 
 * @param R 
 * @param t 
 * @return double 
 */
double PnPsolver::compute_R_and_t(const double *ut, const double *betas, double R[3][3], double t[3])
{
	compute_ccs(betas, ut);//通过Betas和特征向量，得到camera坐标系下的四个控制点 c0,c1,c2,c3
	compute_pcs();

	solve_for_sign();

	estimate_R_and_t(R, t);//3D-3D svd方法求解ICP获得Rt

	return reprojection_error(R, t);
}


/**
 * @brief 	第一个近似解
 * 			除了B00 B01 B02 B03四个参数外的其它参数均为0的最小二乘解，求出B0、B1、B2、B3的粗略解
 * 			betas10		   = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
 *			betas_approx_1 = [B00 B01     B02         B03			 ]
 * 
 * @param L_6x10 
 * @param Rho 
 * @param betas 
 */
void PnPsolver::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho, double *betas)
{
	double l_6x4[6 * 4], b4[4];
	CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
	CvMat B4 = cvMat(4, 1, CV_64F, b4);

	for (int i = 0; i < 6; i++)
	{
		cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
		cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
		cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
		cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
	}

	cvSolve(&L_6x4, Rho, &B4, CV_SVD);

	//B00 = B0*B0 所以一定为正
	if (b4[0] < 0)//如果B00为负，则整体取负
	{

		betas[0] = sqrt(-b4[0]);
		betas[1] = -b4[1] / betas[0];
		betas[2] = -b4[2] / betas[0];
		betas[3] = -b4[3] / betas[0];
	}
	else
	{
		betas[0] = sqrt(b4[0]);
		betas[1] = b4[1] / betas[0];
		betas[2] = b4[2] / betas[0];
		betas[3] = b4[3] / betas[0];
	}
}

/**
 * @brief 第二个近似解
 * 		  除了B00 B01 B11 三个参数外的其它参数均为0的最小二乘解，求出B0、B1、B2、B3的粗略解
 * 		  betas10		 = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
 *		  betas_approx_1 = [B00 B01 B11         				   ]
 * 
 * @param L_6x10 
 * @param Rho 
 * @param betas 
 */
void PnPsolver::find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho,
									double *betas)
{
	double l_6x3[6 * 3], b3[3];
	CvMat L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
	CvMat B3 = cvMat(3, 1, CV_64F, b3);

	for (int i = 0; i < 6; i++)
	{
		cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
		cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
		cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
	}

	cvSolve(&L_6x3, Rho, &B3, CV_SVD);

	//B00,B11都为正，如果不同时为正，则B11为0
	if (b3[0] < 0)
	{
		betas[0] = sqrt(-b3[0]);
		betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
	}
	else
	{
		betas[0] = sqrt(b3[0]);
		betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
	}

	//如果B01为负，则B0，B1异号
	if (b3[1] < 0)
		betas[0] = -betas[0];

	betas[2] = 0.0;
	betas[3] = 0.0;
}


/**
 * @brief 第三个近似解
 * 		  除了B00 B01 B11 B02 B12 五个参数外的其它参数均为0的最小二乘解，求出B0、B1、B2、B3的粗略解
 * 		  betas10		 = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
 *		  betas_approx_1 = [B00 B01 B11 B02 B12   				   ]    
 * 
 * @param L_6x10 
 * @param Rho 
 * @param betas 
 */
void PnPsolver::find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
									double *betas)
{
	double l_6x5[6 * 5], b5[5];
	CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
	CvMat B5 = cvMat(5, 1, CV_64F, b5);

	for (int i = 0; i < 6; i++)
	{
		cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
		cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
		cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
		cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
		cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
	}

	cvSolve(&L_6x5, Rho, &B5, CV_SVD);

	if (b5[0] < 0)
	{
		betas[0] = sqrt(-b5[0]);
		betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
	}
	else
	{
		betas[0] = sqrt(b5[0]);
		betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
	}
	if (b5[1] < 0)
		betas[0] = -betas[0];
	betas[2] = b5[3] / betas[0];
	betas[3] = 0.0;
}

/**
 * @brief 计算L矩阵
 * 
 * @param ut 12x12的特征向量矩阵，ut每一行为一组特征向量解（u的每一列为一组特征向量解）
 *			 
 * @param l_6x10 
 */
void PnPsolver::compute_L_6x10(const double *ut, double *l_6x10)
{
	const double *v[4];//分别取出最优的四组特征向量解，由于svd按照特征值大小降序排列，因此越往下排列的特征向量越优

	v[0] = ut + 12 * 11;
	v[1] = ut + 12 * 10;
	v[2] = ut + 12 * 9;
	v[3] = ut + 12 * 8;

	//4：4个svd中的最优四个特征向量
	//6：四个控制点之间的向量差（距离）：[0,1][0,2][0,3][1,2][1,3][2,3]
	//3：x,y,z三个轴
	double dv[4][6][3];

	for (int i = 0; i < 4; i++)
	{
		int a = 0, b = 1;
		for (int j = 0; j < 6; j++)
		{
			dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
			dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
			dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

			b++;
			if (b > 3)
			{
				a++;
				b = a + 1;
			}
		}
	}
	
	//L 6x10= [dv00,2*dv01,dv11,2*dv02,2*dv12,dv22,2*dv03,2*dv13,2*dv23,dv33]
	for (int i = 0; i < 6; i++)
	{
		double *row = l_6x10 + 10 * i;

		row[0] = dot(dv[0][i], dv[0][i]);
		row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
		row[2] = dot(dv[1][i], dv[1][i]);
		row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
		row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
		row[5] = dot(dv[2][i], dv[2][i]);
		row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
		row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
		row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
		row[9] = dot(dv[3][i], dv[3][i]);
	}
}

/**
 * @brief 计算四个控制点任意两点间的距离，共6个距离
 * 
 * @param rho 
 */
void PnPsolver::compute_rho(double *rho)
{
	rho[0] = dist2(cws[0], cws[1]);
	rho[1] = dist2(cws[0], cws[2]);
	rho[2] = dist2(cws[0], cws[3]);
	rho[3] = dist2(cws[1], cws[2]);
	rho[4] = dist2(cws[1], cws[3]);
	rho[5] = dist2(cws[2], cws[3]);
}

/**
 * @brief 
 * 
 * @param l_6x10 
 * @param rho 
 * @param betas 
 * @param A  6x4
 * @param b  6x1
 */
void PnPsolver::compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
											 double betas[4], CvMat *A, CvMat *b)
{
	for (int i = 0; i < 6; i++)
	{
		const double *rowL = l_6x10 + i * 10;
		double *rowA = A->data.db + i * 4;

		//对B0，B1，B2，B3求雅克比
		// （dv00*B00 + 2*dv01*B01 + dv11*B11 + 2*dv02*B02 + 2*dv12*B12 + dv22*B22 + 2*dv03*B03 + 2*dv13*B13 + 2*dv23*B23 + dv33*B33）
		rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
		rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
		rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
		rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

		//b 为camera下控制点平方距离与世界坐标系下控制点平方距离的残差
		cvmSet(b, i, 0, rho[i] - (rowL[0] * betas[0] * betas[0] + 
								  rowL[1] * betas[0] * betas[1] + 
								  rowL[2] * betas[1] * betas[1] + 
								  rowL[3] * betas[0] * betas[2] + 
								  rowL[4] * betas[1] * betas[2] + 
								  rowL[5] * betas[2] * betas[2] + 
								  rowL[6] * betas[0] * betas[3] + 
								  rowL[7] * betas[1] * betas[3] + 
								  rowL[8] * betas[2] * betas[3] + 
								  rowL[9] * betas[3] * betas[3]));
	}
}

/**
 * @brief 高斯牛顿法优化
 * 
 * @param L_6x10 
 * @param Rho 
 * @param betas 
 */
void PnPsolver::gauss_newton(const CvMat *L_6x10, const CvMat *Rho,double betas[4])
{
	const int iterations_number = 5;//迭代次数

	double a[6 * 4], b[6], x[4];
	CvMat A = cvMat(6, 4, CV_64F, a);
	CvMat B = cvMat(6, 1, CV_64F, b);
	CvMat X = cvMat(4, 1, CV_64F, x);

	for (int k = 0; k < iterations_number; k++)
	{
		//构造Ax = B ，A为目标函数关于待优化变量(B0,B1,B2,B3)的雅克比矩阵
		//B为目标函数当前残差(camera下控制点之间的平方距离与world下控制点之间的平方距离之差)
		compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A, &B);
		qr_solve(&A, &B, &X);

		for (int i = 0; i < 4; i++)
			betas[i] += x[i];
	}
}

/**
 * @brief 列向量线性无关的实矩阵（不一定要求方阵），分解成Q正交阵，R上三角矩阵，
 * 
 * @param A 
 * @param b 
 * @param X 
 */
void PnPsolver::qr_solve(CvMat *A, CvMat *b, CvMat *X)
{
	static int max_nr = 0;
	static double *A1, *A2;

	const int nr = A->rows;
	const int nc = A->cols;

	if (max_nr != 0 && max_nr < nr)
	{
		delete[] A1;
		delete[] A2;
	}
	if (max_nr < nr)
	{
		max_nr = nr;
		A1 = new double[nr];
		A2 = new double[nr];
	}

	double *pA = A->data.db, *ppAkk = pA;
	for (int k = 0; k < nc; k++)
	{
		double *ppAik = ppAkk, eta = fabs(*ppAik);
		for (int i = k + 1; i < nr; i++)
		{
			double elt = fabs(*ppAik);
			if (eta < elt)
				eta = elt;
			ppAik += nc;
		}

		if (eta == 0)
		{
			A1[k] = A2[k] = 0.0;
			cerr << "God damnit, A is singular, this shouldn't happen." << endl;
			return;
		}
		else
		{
			double *ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
			for (int i = k; i < nr; i++)
			{
				*ppAik *= inv_eta;
				sum += *ppAik * *ppAik;
				ppAik += nc;
			}
			double sigma = sqrt(sum);
			if (*ppAkk < 0)
				sigma = -sigma;
			*ppAkk += sigma;
			A1[k] = sigma * *ppAkk;
			A2[k] = -eta * sigma;
			for (int j = k + 1; j < nc; j++)
			{
				double *ppAik = ppAkk, sum = 0;
				for (int i = k; i < nr; i++)
				{
					sum += *ppAik * ppAik[j - k];
					ppAik += nc;
				}
				double tau = sum / A1[k];
				ppAik = ppAkk;
				for (int i = k; i < nr; i++)
				{
					ppAik[j - k] -= tau * *ppAik;
					ppAik += nc;
				}
			}
		}
		ppAkk += nc + 1;
	}

	// b <- Qt b
	double *ppAjj = pA, *pb = b->data.db;
	for (int j = 0; j < nc; j++)
	{
		double *ppAij = ppAjj, tau = 0;
		for (int i = j; i < nr; i++)
		{
			tau += *ppAij * pb[i];
			ppAij += nc;
		}
		tau /= A1[j];
		ppAij = ppAjj;
		for (int i = j; i < nr; i++)
		{
			pb[i] -= tau * *ppAij;
			ppAij += nc;
		}
		ppAjj += nc + 1;
	}

	// X = R-1 b
	double *pX = X->data.db;
	pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
	for (int i = nc - 2; i >= 0; i--)
	{
		double *ppAij = pA + i * nc + (i + 1), sum = 0;

		for (int j = i + 1; j < nc; j++)
		{
			sum += *ppAij * pX[j];
			ppAij++;
		}
		pX[i] = (pb[i] - sum) / A2[i];
	}
}

void PnPsolver::relative_error(double &rot_err, double &transl_err,
							   const double Rtrue[3][3], const double ttrue[3],
							   const double Rest[3][3], const double test[3])
{
	double qtrue[4], qest[4];

	mat_to_quat(Rtrue, qtrue);
	mat_to_quat(Rest, qest);

	double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
						   (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
						   (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
						   (qtrue[3] - qest[3]) * (qtrue[3] - qest[3])) /
					  sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

	double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
						   (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
						   (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
						   (qtrue[3] + qest[3]) * (qtrue[3] + qest[3])) /
					  sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

	rot_err = min(rot_err1, rot_err2);

	transl_err = sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
					  (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
				 	  (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
				 sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

/**
 * @brief rotate matrix to quat
 * 
 * @param R 
 * @param q 
 */
void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
	double tr = R[0][0] + R[1][1] + R[2][2];
	double n4;

	if (tr > 0.0f)
	{
		q[0] = R[1][2] - R[2][1];
		q[1] = R[2][0] - R[0][2];
		q[2] = R[0][1] - R[1][0];
		q[3] = tr + 1.0f;
		n4 = q[3];
	}
	else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2]))
	{
		q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
		q[1] = R[1][0] + R[0][1];
		q[2] = R[2][0] + R[0][2];
		q[3] = R[1][2] - R[2][1];
		n4 = q[0];
	}
	else if (R[1][1] > R[2][2])
	{
		q[0] = R[1][0] + R[0][1];
		q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
		q[2] = R[2][1] + R[1][2];
		q[3] = R[2][0] - R[0][2];
		n4 = q[1];
	}
	else
	{
		q[0] = R[2][0] + R[0][2];
		q[1] = R[2][1] + R[1][2];
		q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
		q[3] = R[0][1] - R[1][0];
		n4 = q[2];
	}
	double scale = 0.5f / double(sqrt(n4));

	q[0] *= scale;
	q[1] *= scale;
	q[2] *= scale;
	q[3] *= scale;
}

} // namespace ORB_SLAM2
