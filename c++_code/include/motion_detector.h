#ifndef LIGHT_DT_MOTION_DETECTOR_H
#define LIGHT_DT_MOTION_DETECTOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class MotionDetector {
public:
    MotionDetector(const std::string& model_path);
    ~MotionDetector() = default;

    cv::Rect detect(const cv::Mat& prevFrame, const cv::Mat& currentFrame);

private:
    Ort::Env env_;
    Ort::Session session_;

    static constexpr int CLASSIFY_SIZE = 32;

    struct CompensateResult {
        cv::Mat compensated;
        cv::Mat mask;
        double avg_dist;
    };

    CompensateResult motionCompensate(const cv::Mat& gray1, const cv::Mat& gray2);
    int classify(const cv::Mat& crop);
    cv::Rect enlargeBox(int x, int y, int w, int h, int pad, int img_w, int img_h);
};

#endif // LIGHT_DT_MOTION_DETECTOR_H
