#ifndef LIGHT_DT_LIGHT_TRACK_H
#define LIGHT_DT_LIGHT_TRACK_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Bbox {
    float x0;
    float y0;
    float x1;
    float y1;
};

class LightTrack {
public:
    LightTrack(const std::string& model_init, const std::string& model_update);
    ~LightTrack() = default;

    void init(const uint8_t *img, Bbox &box, int im_h, int im_w);
    void track(const uint8_t *img);

    int cls_score_position = 0;
    float track_score = 0.0f;
    bool target_pos_change();

    cv::Point target_pos = {0, 0};
    cv::Point2f target_sz = {0.f, 0.f};

private:
    Ort::Env env_;
    Ort::Session session_init_;
    Ort::Session session_update_;

    std::vector<float> zf_;
    static constexpr int ZF_CHANNELS = 96;
    static constexpr int ZF_SIZE = 8;

    int stride = 16;
    int exemplar_size = 127;
    int instance_size = 288;
    float lr = 0.616f;
    float penalty_tk = 0.007f;
    float context_amount = 0.5f;
    float window_influence = 0.225f;
    int score_size;
    int total_stride = 16;

    int ori_img_w = 960;
    int ori_img_h = 640;

    static constexpr float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float std_vals[3]  = {0.229f, 0.224f, 0.225f};

    void grids();
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz);
    void update(const cv::Mat &x_crops, float scale_z);
    std::vector<float> preprocess(const cv::Mat& img);

    std::vector<float> window;
    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
};

#endif // LIGHT_DT_LIGHT_TRACK_H
