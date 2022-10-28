#include "WeightedMovingVariance.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

using namespace bgslibrary::algorithms;

WeightedMovingVariance::WeightedMovingVariance() 
    : m_params(true, true, 15),
    m_numProcessesParallel(12)
{
    imgInputPrevParallel.resize(m_numProcessesParallel);

    for (int i = 0; i < m_numProcessesParallel; ++i) {
        m_processSeq.push_back(i);
        imgInputPrevParallel[i][0] = std::make_unique<cv::Mat>();
        imgInputPrevParallel[i][1] = std::make_unique<cv::Mat>();
    }
}

WeightedMovingVariance::~WeightedMovingVariance()
{
}

void WeightedMovingVariance::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput)
{
    if (m_numProcessesParallel > 1) 
        processParallel(_imgInput, _imgOutput);
    else
        process(_imgInput, _imgOutput, imgInputPrevParallel[0], m_params);
}

void WeightedMovingVariance::processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput) {
    std::for_each(
        std::execution::par,
        m_processSeq.begin(),
        m_processSeq.end(),
        [&](int np)
        {
            int height = _imgInput.size().height / m_numProcessesParallel;
            int pixelPos = np * _imgInput.size().width * height;
            cv::Mat imgSplit(height, _imgInput.size().width, _imgInput.type(), _imgInput.data + (pixelPos * _imgInput.channels()));
            cv::Mat maskPartial(height, _imgInput.size().width, _imgOutput.type(), _imgOutput.data + pixelPos);
            process(imgSplit, maskPartial, imgInputPrevParallel[np], m_params);
        });
}

void WeightedMovingVariance::process(const cv::Mat &img_input, 
                                    cv::Mat &img_output, 
                                    std::array<std::unique_ptr<cv::Mat>, 2>& imgInputPrev, 
                                    const WeightedMovingVarianceParams& _params)
{
    static const float oneThird = 1.0f / 3.0f;

    auto img_input_f = std::make_unique<cv::Mat>();
    img_input.convertTo(*img_input_f, CV_32F, 1. / 255.);

    if (imgInputPrev[0]->empty())
    {
        imgInputPrev[0] = std::move(img_input_f);
        return;
    }

    if (imgInputPrev[1]->empty())
    {
        imgInputPrev[1] = std::move(imgInputPrev[0]);
        imgInputPrev[0] = std::move(img_input_f);
        return;
    }

    cv::Mat& img_input_prev_1_f = *imgInputPrev[0];
    cv::Mat& img_input_prev_2_f = *imgInputPrev[1];

    // Weighted variance
    cv::Mat imgProcess;

    if (_params.enableWeight) {
        // Weighted mean
        computeWeightedVarianceCombined(*img_input_f, img_input_prev_1_f, img_input_prev_2_f, 
                                        0.5f, 0.3f, 0.2f, imgProcess);
    } else {
        // Weighted mean
        computeWeightedVarianceCombined(*img_input_f, img_input_prev_1_f, img_input_prev_2_f, 
                                        oneThird, oneThird, oneThird, imgProcess);
    }

    if (imgProcess.channels() == 3)
        cv::cvtColor(imgProcess, imgProcess, CV_BGR2GRAY);

    if (_params.enableThreshold)
         cv::threshold(imgProcess, imgProcess, _params.threshold, 255, cv::THRESH_BINARY);

    memcpy(img_output.data, imgProcess.data, img_output.size().width * img_output.size().height);

    imgInputPrev[1] = std::move(imgInputPrev[0]);
    imgInputPrev[0] = std::move(img_input_f);
}

void WeightedMovingVariance::computeWeightedVarianceCombined(
        const cv::Mat &img1F, 
        const cv::Mat &img2F, 
        const cv::Mat &img3F, 
        const float weight1, 
        const float weight2, 
        const float weight3, 
        cv::Mat& img_f)
{
    img_f.create(img1F.size(), CV_8UC(img1F.channels()));
    size_t totalDataSize{img1F.size().area() * (size_t)img1F.channels()};
    float *dataI1 = (float*)img1F.data;
    float *dataI2 = (float*)img2F.data;
    float *dataI3 = (float*)img3F.data;
    uchar *dataOut = img_f.data;
    for (size_t i{0}; i < totalDataSize; ++i) {
        const float mean{(*dataI1 * weight1) + (*dataI2 * weight2) + (*dataI3 * weight3)};
        const float value1{std::abs(*dataI1 - mean)};
        const float value2{std::abs(*dataI2 - mean)};
        const float value3{std::abs(*dataI3 - mean)};
        const float result = std::sqrt(((value1 * value1) * weight1) + ((value2 * value2) * weight2) + ((value3 * value3) * weight3));
        *dataOut = (uchar)(result * 255.0f);
        ++dataOut;
        ++dataI1;
        ++dataI2;
        ++dataI3;
    }
}
