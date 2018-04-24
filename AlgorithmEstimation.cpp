#include "AlgorithmEstimation.hpp"
#include <fstream>
#include <iterator>
#include <cstdint>

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;
    
    std::vector<float> distances(matches.size());
    for (size_t i=0; i<matches.size(); i++)
        distances[i] = matches[i].distance;
    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);
    
    meanDistance = static_cast<float>(mean.val[0]);
    stdDev       = static_cast<float>(dev.val[0]);
    
    return false;
}

float distance(const cv::Point2f a, const cv::Point2f b)
{
    return sqrt((a - b).dot(a-b));
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);


bool performEstimation
(
 const FeatureAlgorithm& alg,//第一个参数 传入算法名称
 const ImageTransformation& transformation,//第二个参数 传入变换名称
 const cv::Mat& sourceImage,
 std::vector<FrameMatchingStatistics>& stat
)
{
    Keypoints   sourceKp;
    Descriptors sourceDesc;

    cv::Mat gray;

    if (sourceImage.channels() == 3)
    {
        cv::cvtColor(sourceImage, gray, cv::COLOR_BGR2GRAY);
    }
    else if (sourceImage.channels() == 4)
    {
        cv::cvtColor(sourceImage, gray, cv::COLOR_BGRA2GRAY);
    }
    else if(sourceImage.channels() == 1)
    {
        gray = sourceImage;
    }

    if (!alg.extractFeatures(gray, sourceKp, sourceDesc))//提取原图的灰度图的特征点和描述子
        return false;
    
    std::vector<float> x = transformation.getX();//变换的参数组成的数组
    stat.resize(x.size());
    
    const int count = x.size();
    
    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches,inliermatches;
    
    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();//毫秒为单位
    
    //#pragma omp parallel for private(resKpReal, resDesc, matches) schedule(dynamic, 5)
    for (int i = 0; i < count; i++)
    {
        float       arg = x[i];
        FrameMatchingStatistics& s = stat[i];//s类包含了配准效果的各种参数
        
        cv::Mat     transformedImage;
        transformation.transform(arg, gray, transformedImage);//通过重载得到不同变换后的图像

        if (0)
        {
            std::ostringstream image_name;
            image_name << "image_dump_" << transformation.name << "_" << i << ".bin";
            std::ofstream dump(image_name.str().c_str(), std::ios::binary);
            std::copy(transformedImage.datastart, transformedImage.dataend, std::ostream_iterator<uint8_t>(dump));
        }
        cv::Mat expectedHomography = transformation.getHomography(arg, gray);//????直接看定义只返回一个单位阵，但是只是初始化也会重载
                
        int64 start = cv::getTickCount();
        
        alg.extractFeatures(transformedImage, resKpReal, resDesc);//提取变换后图像的特征点和描述子

        // Initialize required fields
        s.isValid        = resKpReal.size() > 0;
        s.argumentValue  = arg;
        
        if (!s.isValid)
            continue;

        alg.matchFeatures(sourceDesc, resDesc, matches);//由两幅图的描述子train和query得到匹配对
		//这里可以用knn优化 但不能直接用rob Hess的代码，他的代码是c语言写的

        int64 end = cv::getTickCount();

        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert(sourceKp, sourcePoints);// sourceKp是Keypoints类型的 转换为Point2f
        cv::perspectiveTransform(sourcePoints, sourcePointsInFrame, expectedHomography);//把原图的特征点集通过矩阵变换到另外一组，对应理想变换后的
		//you want to compute the most probable perspective transformation out of several
			/*pairs of corresponding points, you can use getPerspectiveTransform or
			findHomography .*/
        cv::Mat homography;

        //so, we have :
        //N - number of keypoints in the first image that are also visible
        //    (after transformation) on the second image

        //    N1 - number of keypoints in the first image that have been matched.

        //    n - number of the correct matches found by the matcher

        //    n / N1 - precision
        //    n / N - recall(? )

        int visibleFeatures = 0;
        int correctMatches  = 0;
        int matchesCount    = matches.size();

        for (int i = 0; i < sourcePoints.size(); i++)
        {
            if (sourcePointsInFrame[i].x > 0 &&
                sourcePointsInFrame[i].y > 0 &&
                sourcePointsInFrame[i].x < transformedImage.cols &&
                sourcePointsInFrame[i].y < transformedImage.rows)
            {
                visibleFeatures++;//原图计算特征点理想变换后可以落在图像区域内的点数
            }
        }

        for (int i = 0; i < matches.size(); i++)
        {
            cv::Point2f expected = sourcePointsInFrame[matches[i].trainIdx];////trainIdx为train描述子的索引，match函数中后面的那个描述子 
            cv::Point2f actual   = resKpReal[matches[i].queryIdx].pt;// //queryIdx为query描述子的索引，match函数中前面的那个描述子  //这里感觉应该也是trainIdx索引，最起码也是都可以
            
            if (distance(expected, actual) < 3.0)
            {
                correctMatches++;//预期点和变换后图像匹配后的特征点距离小于3就认为是正确匹配
            }
        }

       // bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, correctMatches, homography);
		bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, inliermatches, homography);//这里本来是注释掉的，直接用会报错  求出的是第一个参数变换到第二个的变换矩阵
		//可选用least-median或者RANSAC计算出内点对和估计出的变换矩阵，返回一个bool值
		//inlinermatches是为了计算变换矩阵的内点，与刚才判断的是否是正确匹配对不牵扯

        // Some simple stat:
        s.isValid        = homographyFound;
        s.totalKeypoints = resKpReal.size();
        s.consumedTimeMs = (end - start) * toMsMul;//匹配耗时
        s.precision = correctMatches / (float) matchesCount;
        s.recall = correctMatches / (float) visibleFeatures;//正确匹配对数占可见特征点的比例

		s.correctMatchesPercent = s.precision;
		s.percentOfMatches = (float)matchesCount / (s.totalKeypoints);
		//matchingRatio再通过以上二者的乘积计算 要不然match都是0。matchingRatio()     const { return correctMatchesPercent * percentOfMatches * 100.0f; };

		
        
        // Compute matching statistics
        //if (homographyFound)
        //{
        //    cv::Mat r = expectedHomography * homography.inv();
        //    float error = cv::norm(cv::Mat::eye(3,3, CV_64FC1) - r, cv::NORM_INF);

        //    computeMatchesDistanceStatistics(correctMatches, s.meanDistance, s.stdDevDistance);
        //    s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, correctMatches, homography);
        //    s.homographyError = std::min(error, 1.0f);

        //    if (0 && error >= 1)
        //    {
        //        std::cout << "H expected:" << expectedHomography << std::endl;
        //        std::cout << "H actual:"   << homography << std::endl;
        //        std::cout << "H error:"    << error << std::endl;
        //        std::cout << "R error:"    << s.reprojectionError(0) << ";" 
        //                                   << s.reprojectionError(1) << ";" 
        //                                   << s.reprojectionError(2) << ";" 
        //                                   << s.reprojectionError(3) << std::endl;
        //        
        //        cv::Mat matchesImg;
        //        cv::drawMatches(transformedImage,
        //                        resKpReal,
        //                        gray,
        //                        sourceKp,
        //                        correctMatches,
        //                        matchesImg,
        //                        cv::Scalar::all(-1),
        //                        cv::Scalar::all(-1),
        //                        std::vector<char>(),
        //                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //        
        //        cv::imshow("Matches", matchesImg);
        //        cv::waitKey(-1);
        //    }
        //}

		// Compute matching statistics 这部分一开始也被注释掉了
		if (homographyFound)
		{
			cv::Mat r = expectedHomography * homography.inv();//矩阵求逆,理想情况正好得到一个单位阵
			float error = cv::norm(cv::Mat::eye(3, 3, CV_64FC1) - r, cv::NORM_INF);//求无穷大范数 得到的误差以像素为单位 norm可重载 矩阵元素绝对值的平方和再开平方
			// 知乎上 使用opencv如何进行两个特征点集之间的相似度比较的？第一个答案就是用范数
			//两个数很好进行大小比较，但向量不好怎么比较，于是引入了范数

			computeMatchesDistanceStatistics(inliermatches, s.meanDistance, s.stdDevDistance);//计算配对对之间的距离均值和标准差 但是这里的匹配对是RANSAC之前还是之后呢？
			s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, matches, homography);//得到配对对的距离均值，标准差 最大值 最小值组成的scalar
			s.homographyError = std::min(error, 1.0f);//直接1.0是double型，后面加f变成float型的
			//只存放小于1的错误

			//计算psnr
			cv::Mat srcImage = transformedImage.clone();
			cv::Mat dstImage = sourceImage.clone();
			//cv::perspectiveTransform(srcImage, dstImage,homography.inv()); //将待配准图像按照求出的矩阵配准
			//输入并不是图像，而是图像对应的坐标
			cv::warpPerspective(srcImage, dstImage, homography.inv(), dstImage.size(), CV_INTER_CUBIC);
			s.psnr = PSNR(dstImage, gray);//返回的是double类型 要求输入是灰度图像
			//求出的psnr怎么一直是0??还中断。。

			

			if (0 && error >= 1)//0&&??
			{
				std::cout << "H expected:" << expectedHomography << std::endl;
				std::cout << "H actual:" << homography << std::endl;
				std::cout << "H error:" << error << std::endl;
				std::cout << "R error:" << s.reprojectionError(0) << ";"
					<< s.reprojectionError(1) << ";"
					<< s.reprojectionError(2) << ";"
					<< s.reprojectionError(3) << std::endl;

				cv::Mat matchesImg;
				cv::drawMatches(transformedImage,
					resKpReal,
					gray,
					sourceKp,
					inliermatches,
					matchesImg,
					cv::Scalar::all(-1),
					cv::Scalar::all(-1),
					std::vector<char>(),
					cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				//参数里面没有int型的correctmatches http://blog.sina.com.cn/sblog_a98e39a201017pgn.html
				cv::imshow("Matches", matchesImg);
				cv::waitKey(-1);
			}
		}

    }
    
    return true;
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography)
{
    assert(matches.size() > 0);

    const int pointsCount = matches.size();
    std::vector<cv::Point2f> srcPoints, dstPoints;
    std::vector<float> distances;

    for (int i = 0; i < pointsCount; i++)
    {
        srcPoints.push_back(source[matches[i].trainIdx].pt);//把匹配对的特征点压入srcPoints中
        dstPoints.push_back(query[matches[i].queryIdx].pt);// //trainIdx为train描述子的索引，match函数中后面的那个描述子  
    }

    cv::perspectiveTransform(dstPoints, dstPoints, homography.inv());//利用求出的变换矩阵把特征点变换到一个坐标系中，要想配准，就要求逆  输入并不是图像，而是图像对应的坐标
    for (int i = 0; i < pointsCount; i++)
    {
        const cv::Point2f& src = srcPoints[i];
        const cv::Point2f& dst = dstPoints[i];

        cv::Point2f v = src - dst;//匹配对中的特征点之差
        distances.push_back(sqrtf(v.dot(v)));//算出每一对特征点的距离，是为了计算衡量配准的标准，而不是找对应点
    }

    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);

    cv::Scalar result;
    result(0) = mean(0);
    result(1) = dev(0);
    result(2) = *std::max_element(distances.begin(), distances.end());
    result(3) = *std::min_element(distances.begin(), distances.end());
    return result;
}
