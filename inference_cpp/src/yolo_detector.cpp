#include "yolo_detector.h"
#include <algorithm>
#include <numeric>
#include <iostream>

YoloDetector::YoloDetector(const std::string& model_path, int input_w, int input_h)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector")
    , session_(nullptr)
    , input_w_(input_w)
    , input_h_(input_h)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "ONNX Runtime: CUDA provider enabled" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime: CUDA unavailable (" << e.what() << "), using CPU" << std::endl;
    }

    session_ = Ort::Session(env_, model_path.c_str(), opts);

    auto input_name = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name.get();

    auto output_name = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name.get();

    std::cout << "ONNX model loaded: " << model_path << std::endl;
    std::cout << "  Input:  " << input_name_ << " [1, 3, " << input_h_ << ", " << input_w_ << "]" << std::endl;
    std::cout << "  Output: " << output_name_ << std::endl;
}

cv::Mat YoloDetector::letterbox(const cv::Mat& img, float& ratio, int& pad_x, int& pad_y) {
    float r_w = (float)input_w_ / img.cols;
    float r_h = (float)input_h_ / img.rows;
    ratio = std::min(r_w, r_h);

    int new_w = static_cast<int>(img.cols * ratio);
    int new_h = static_cast<int>(img.rows * ratio);
    pad_x = (input_w_ - new_w) / 2;
    pad_y = (input_h_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return out;
}

float YoloDetector::iou(const Detection& a, const Detection& b) {
    float a_x1 = a.bbox[0] - a.bbox[2] / 2, a_y1 = a.bbox[1] - a.bbox[3] / 2;
    float a_x2 = a.bbox[0] + a.bbox[2] / 2, a_y2 = a.bbox[1] + a.bbox[3] / 2;
    float b_x1 = b.bbox[0] - b.bbox[2] / 2, b_y1 = b.bbox[1] - b.bbox[3] / 2;
    float b_x2 = b.bbox[0] + b.bbox[2] / 2, b_y2 = b.bbox[1] + b.bbox[3] / 2;

    float inter_x1 = std::max(a_x1, b_x1), inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2), inter_y2 = std::min(a_y2, b_y2);

    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) return 0.0f;

    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float union_area = a.bbox[2] * a.bbox[3] + b.bbox[2] * b.bbox[3] - inter_area;
    return inter_area / union_area;
}

void YoloDetector::nms(std::vector<Detection>& dets, float nms_thresh) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    std::vector<Detection> result;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].class_id == dets[j].class_id &&
                iou(dets[i], dets[j]) > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }
    dets = std::move(result);
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame, float conf_thresh, float nms_thresh) {
    // Letterbox preprocessing
    float ratio;
    int pad_x, pad_y;
    cv::Mat input_img = letterbox(frame, ratio, pad_x, pad_y);

    // BGR -> RGB, HWC -> CHW, normalize to [0, 1]
    cv::Mat rgb;
    cv::cvtColor(input_img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    std::vector<float> input_data(3 * input_h_ * input_w_);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data.data() + c * input_h_ * input_w_,
               channels[c].data, input_h_ * input_w_ * sizeof(float));
    }

    // Run inference
    std::array<int64_t, 4> input_shape = {1, 3, input_h_, input_w_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1, output_names, 1);

    // Parse output [1, 25200, 6]: cx, cy, w, h, obj_conf, cls_score
    auto& out_tensor = output_tensors[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    int num_preds = static_cast<int>(out_shape[1]);
    int pred_dim = static_cast<int>(out_shape[2]);
    int num_classes = pred_dim - 5;
    const float* out_data = out_tensor.GetTensorData<float>();

    std::vector<Detection> detections;
    for (int i = 0; i < num_preds; ++i) {
        const float* row = out_data + i * pred_dim;
        float obj_conf = row[4];
        if (obj_conf < conf_thresh) continue;

        int best_cls = 0;
        float best_cls_score = row[5];
        for (int c = 1; c < num_classes; ++c) {
            if (row[5 + c] > best_cls_score) {
                best_cls_score = row[5 + c];
                best_cls = c;
            }
        }

        float final_conf = obj_conf * best_cls_score;
        if (final_conf < conf_thresh) continue;

        // Map from letterbox coords back to original image coords
        Detection det;
        det.bbox[0] = (row[0] - pad_x) / ratio;
        det.bbox[1] = (row[1] - pad_y) / ratio;
        det.bbox[2] = row[2] / ratio;
        det.bbox[3] = row[3] / ratio;
        det.conf = final_conf;
        det.class_id = best_cls;
        detections.push_back(det);
    }

    nms(detections, nms_thresh);
    return detections;
}

cv::Rect YoloDetector::get_rect(const cv::Mat& img, const float bbox[4]) {
    int x = std::max(0, static_cast<int>(bbox[0] - bbox[2] / 2));
    int y = std::max(0, static_cast<int>(bbox[1] - bbox[3] / 2));
    int w = std::min(static_cast<int>(bbox[2]), img.cols - x);
    int h = std::min(static_cast<int>(bbox[3]), img.rows - y);
    return cv::Rect(x, y, w, h);
}
