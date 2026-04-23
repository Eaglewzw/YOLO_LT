#ifndef LIGHT_DT_YOLO_DETECTOR_H
#define LIGHT_DT_YOLO_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    float bbox[4]; // cx, cy, w, h (in original image coords)
    float conf;
    int class_id;
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path, int input_w = 640, int input_h = 640);

    std::vector<Detection> detect(const cv::Mat& frame, float conf_thresh = 0.5f, float nms_thresh = 0.4f);

    cv::Rect get_rect(const cv::Mat& img, const float bbox[4]);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    int input_w_;
    int input_h_;
    std::string input_name_;
    std::string output_name_;

    cv::Mat letterbox(const cv::Mat& img, float& ratio, int& pad_x, int& pad_y);
    void nms(std::vector<Detection>& dets, float nms_thresh);
    static float iou(const Detection& a, const Detection& b);
};

#endif // LIGHT_DT_YOLO_DETECTOR_H
