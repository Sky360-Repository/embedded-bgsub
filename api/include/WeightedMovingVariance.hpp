#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace bgslibrary
{
    namespace algorithms
    {
        struct WeightedMovingVarianceParams
        {
            WeightedMovingVarianceParams(bool _enableWeight,
                                        bool _enableThreshold,
                                        int _threshold)
                : enableWeight(_enableWeight),
                enableThreshold(_enableThreshold),
                threshold(_threshold)
            {}

            const bool enableWeight;
            const bool enableThreshold;
            const int threshold;
        };

        class WeightedMovingVariance
        {
        public:
            WeightedMovingVariance();
            ~WeightedMovingVariance();

            void process(const cv::Mat &img_input, cv::Mat &img_output);

        private:
            static inline const float ONE_THIRD{1.0f / 3.0f};

            const int m_numProcessesParallel;
            std::vector<int> m_processSeq;
            std::vector<std::array<std::unique_ptr<cv::Mat>, 2>> imgInputPrevParallel;

            const WeightedMovingVarianceParams m_params;

            void processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput);
            static void process(const cv::Mat &img_input, 
                                cv::Mat &img_output, 
                                std::array<std::unique_ptr<cv::Mat>, 2>& img_input_prev, 
                                const WeightedMovingVarianceParams& _params);
            static void weightedVarianceMono(
                    const cv::Mat &_img1, 
                    const cv::Mat &_img2, 
                    const cv::Mat &_img3, 
                    cv::Mat& _outImg,
                    const WeightedMovingVarianceParams& _params);
            static void weightedVarianceColor(
                    const cv::Mat &_img1, 
                    const cv::Mat &_img2, 
                    const cv::Mat &_img3, 
                    cv::Mat& _outImg,
                    const WeightedMovingVarianceParams& _params);
        };
    }
}
