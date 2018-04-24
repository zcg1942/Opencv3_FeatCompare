
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <numeric>
#include <fstream>
	
#include <opencv2\features2d\features2d.hpp> 
#include <opencv2\xfeatures2d.hpp>


const bool USE_VERBOSE_TRANSFORMATIONS = false;

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useBF = true;

    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("ORB",   cv::ORB::create(),   useBF));
    algorithms.push_back(FeatureAlgorithm("AKAZE", cv::AKAZE::create(), useBF));//AKAZE继承于Feature2D
    algorithms.push_back(FeatureAlgorithm("KAZE",  cv::KAZE::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("BRISK", cv::BRISK::create(), useBF));
    algorithms.push_back(FeatureAlgorithm("SURF",  cv::xfeatures2d::SURF::create(),  useBF));//surf,sift在命名空间xfeatures2d中
	algorithms.push_back(FeatureAlgorithm("SIFT", cv::xfeatures2d::SIFT::create(), useBF));
	
	//algorithms.push_back(FeatureAlgorithm("FREAK", cv::xfeatures2d::FREAK::create(), useBF));
   /* algorithms.push_back(FeatureAlgorithm("FREAK", cv::fPtr<cv::FeatureDetector>(new cv::SurfFeatureDetector(2000,4)), cv::Ptr<cv::DescriptorExtractor>(new cv::FREAK()), useBF));*/

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 1)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.01f)));
    }
    else
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.1f)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 10)));//变化范围和步长
    }

    if (argc < 2)
    {
        std::cout << "At least one input image should be passed" << std::endl;
    }

    for (int imageIndex = 1; imageIndex < argc; imageIndex++)
    {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);

        CollectedStatistics fullStat;

        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            std::cout << "Testing " << alg.name << "...";

            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
            {
                const ImageTransformation& trans = *transformations[transformIndex].get();

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));
				//返回一个bool类型，在这个过程中就完成了图像配准
            }

            std::cout << "done." << std::endl;
        }

		//将测试的平均值打印在控制台中
		fullStat.printAverage(std::cout, StatisticsElementRecall);
		fullStat.printAverage(std::cout, StatisticsElementPrecision);

		fullStat.printAverage(std::cout, StatisticsElementHomographyError);
		fullStat.printAverage(std::cout, StatisticsElementMatchingRatio);
		fullStat.printAverage(std::cout, StatisticsElementMeanDistance);
		fullStat.printAverage(std::cout, StatisticsElementPercentOfCorrectMatches);
		fullStat.printAverage(std::cout, StatisticsElementPercentOfMatches);
		fullStat.printAverage(std::cout, StatisticsElementPointsCount);

		fullStat.printPerformanceStatistics(std::cout);

		//将测试的各个值保存在txt中，再用matlab绘图
		std::ofstream recallLog("Recall.txt");//文件写操作并关联文件 内存写入存储设备   
		fullStat.printStatistics(recallLog, StatisticsElementRecall);
		std::ofstream precisionLog("Precision.txt");
		fullStat.printStatistics(precisionLog, StatisticsElementPrecision);

		//补充psnr
		std::ofstream PSNRLog("Psnr.txt");
		fullStat.printStatistics(PSNRLog, StatisticsElementPsnr);

		std::ofstream HomographyErrorLog(" HomographyError.txt ");
		fullStat.printStatistics(HomographyErrorLog, StatisticsElementHomographyError);
		std::ofstream MatchingRatioLog("MatchingRatio.txt");
		fullStat.printStatistics(MatchingRatioLog, StatisticsElementMatchingRatio);
		std::ofstream MeanDistanceLog("MeanDistance.txt");
		fullStat.printStatistics(MeanDistanceLog, StatisticsElementMeanDistance);
		std::ofstream PercentOfCorrectMatchesLog("PercentOfCorrectMatches.txt  ");
		fullStat.printStatistics(PercentOfCorrectMatchesLog, StatisticsElementPercentOfCorrectMatches);
		std::ofstream PercentOfMatchesLog("PercentOfMatches.txt");
		fullStat.printStatistics(PercentOfMatchesLog, StatisticsElementPercentOfMatches);
		std::ofstream PerformanceLog("Performance.txt");
		fullStat.printPerformanceStatistics(PerformanceLog);
    }
	getchar();

    return 0;
}

