#ifndef MOTIONDETECTOR_H
#define MOTIONDETECTOR_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "net.h"

class MotionDetector
{
    private:
        ncnn::Net MobileNetV2;


    public:
        MotionDetector(/* args */);
        cv::Rect MotionDiff(cv::Mat prevFrame, cv::Mat currentFrame);
        float MobileNetV2_Infer(const cv::Mat& bgr);
        ~MotionDetector();
};




#endif



