#include "light_track.h"
#include <algorithm>
#include <cmath>
#include <iostream>

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh) {
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h, float sz) {
    int rows = int(std::sqrt(h.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pad[i * cols + j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float t = std::sqrt((w[i * cols + j] + pad[i * cols + j]) * (h[i * cols + j] + pad[i * cols + j])) / sz;
            sz2.push_back(std::max(t, 1.0f / t));
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz) {
    int rows = int(std::sqrt(h.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, 1.0f / t));
        }
    }
    return sz2;
}

// ==================== LightTrack ====================

LightTrack::LightTrack(const std::string& model_init, const std::string& model_update)
    : env_(ORT_LOGGING_LEVEL_WARNING, "LightTrack")
    , session_init_(nullptr)
    , session_update_(nullptr)
{
    score_size = int(round(instance_size / total_stride));

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "LightTrack: CUDA provider enabled" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "LightTrack: CUDA unavailable, using CPU" << std::endl;
    }

    std::string init_path = model_init + ".onnx";
    std::string update_path = model_update + ".onnx";

    session_init_ = Ort::Session(env_, init_path.c_str(), opts);
    session_update_ = Ort::Session(env_, update_path.c_str(), opts);

    std::cout << "LightTrack models loaded:" << std::endl;
    std::cout << "  Init:   " << init_path << std::endl;
    std::cout << "  Update: " << update_path << std::endl;
}

std::vector<float> LightTrack::preprocess(const cv::Mat& img) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    int h = rgb.rows, w = rgb.cols;
    std::vector<float> data(3 * h * w);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; c++) {
        float* dst = data.data() + c * h * w;
        const float* src = (const float*)channels[c].data;
        for (int i = 0; i < h * w; i++) {
            dst[i] = (src[i] - mean_vals[c]) / std_vals[c];
        }
    }
    return data;
}

void LightTrack::init(const uint8_t *img, Bbox &box, int im_h, int im_w) {
    ori_img_h = im_h;
    ori_img_w = im_w;

    target_sz.x = box.x1 - box.x0;
    target_sz.y = box.y1 - box.y0;
    target_pos.x = box.x0 + (box.x1 - box.x0) / 2;
    target_pos.y = box.y0 + (box.y1 - box.y0) / 2;

    std::cout << "init target pos: " << target_pos << std::endl;
    std::cout << "init target_sz: " << target_sz << std::endl;

    grids();

    float wc_z = target_sz.x + context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + context_amount * (target_sz.x + target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));

    cv::Mat img_(im_h, im_w, CV_8UC3, (void*)img, im_w * 3);
    cv::Mat z_crop = get_subwindow_tracking(img_, target_pos, exemplar_size, int(s_z));

    // Run init model: [1, 3, 127, 127] -> [1, 96, 8, 8]
    std::vector<float> input_data = preprocess(z_crop);
    std::array<int64_t, 4> input_shape = {1, 3, exemplar_size, exemplar_size};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"input1"};
    const char* output_names[] = {"output.1"};

    auto outputs = session_init_.Run(Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1, output_names, 1);

    // Store zf feature map
    const float* zf_data = outputs[0].GetTensorData<float>();
    size_t zf_size = ZF_CHANNELS * ZF_SIZE * ZF_SIZE;
    zf_.assign(zf_data, zf_data + zf_size);

    // Create hanning window
    std::vector<float> hanning(score_size, 0);
    window.resize(score_size * score_size);
    for (int i = 0; i < score_size; i++) {
        hanning[i] = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
    }
    for (int i = 0; i < score_size; i++) {
        for (int j = 0; j < score_size; j++) {
            window[i * score_size + j] = hanning[i] * hanning[j];
        }
    }
}

static int last_cls_score_positon = 170;
bool LightTrack::target_pos_change() {
    bool changed = false;
    if (abs(cls_score_position - last_cls_score_positon) <= 25) {
        last_cls_score_positon = cls_score_position;
    } else {
        changed = true;
        last_cls_score_positon = 170;
    }
    return changed;
}

void LightTrack::update(const cv::Mat &x_crops, float scale_z) {
    // Preprocess search image
    std::vector<float> search_data = preprocess(x_crops);
    std::array<int64_t, 4> search_shape = {1, 3, instance_size, instance_size};

    // zf tensor
    std::array<int64_t, 4> zf_shape = {1, ZF_CHANNELS, ZF_SIZE, ZF_SIZE};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, zf_.data(), zf_.size(), zf_shape.data(), zf_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, search_data.data(), search_data.size(), search_shape.data(), search_shape.size()));

    const char* input_names[] = {"input1", "input2"};
    const char* output_names[] = {"output.1", "output.2"};

    auto outputs = session_update_.Run(Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 2, output_names, 2);

    // Parse cls_score [1, 1, 18, 18]
    const float* cls_score_data = outputs[0].GetTensorData<float>();
    int cols = score_size;  // 18
    int rows = score_size;  // 18

    std::vector<float> cls_score_sigmoid(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        cls_score_sigmoid[i] = sigmoid(cls_score_data[i]);
    }

    auto max_it = std::max_element(cls_score_sigmoid.begin(), cls_score_sigmoid.end());
    cls_score_position = std::distance(cls_score_sigmoid.begin(), max_it);
    track_score = *max_it;

    // Parse bbox_pred [1, 4, 18, 18]
    const float* bbox_pred_data = outputs[1].GetTensorData<float>();
    int spatial = rows * cols;  // 324

    const float* bbox_ch0 = bbox_pred_data;
    const float* bbox_ch1 = bbox_pred_data + spatial;
    const float* bbox_ch2 = bbox_pred_data + spatial * 2;
    const float* bbox_ch3 = bbox_pred_data + spatial * 3;

    std::vector<float> pred_x1(spatial), pred_y1(spatial), pred_x2(spatial), pred_y2(spatial);
    for (int i = 0; i < spatial; i++) {
        pred_x1[i] = grid_to_search_x[i] - bbox_ch0[i];
        pred_y1[i] = grid_to_search_y[i] - bbox_ch1[i];
        pred_x2[i] = grid_to_search_x[i] + bbox_ch2[i];
        pred_y2[i] = grid_to_search_y[i] + bbox_ch3[i];
    }

    // Size penalty
    std::vector<float> w(spatial), h(spatial);
    for (int i = 0; i < spatial; i++) {
        w[i] = pred_x2[i] - pred_x1[i];
        h[i] = pred_y2[i] - pred_y1[i];
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(spatial);
    for (int i = 0; i < spatial; i++) {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i] - 1) * penalty_tk);
    }

    // Window penalty
    std::vector<float> pscore(spatial);
    int r_max = 0, c_max = 0;
    float maxScore = 0;
    for (int i = 0; i < spatial; i++) {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - window_influence) + window[i] * window_influence;
        if (pscore[i] > maxScore) {
            maxScore = pscore[i];
            r_max = std::floor(i / rows);
            c_max = ((float)i / rows - r_max) * rows;
        }
    }

    // Decode to real size
    int idx = r_max * cols + c_max;
    float pred_xs = (pred_x1[idx] + pred_x2[idx]) / 2;
    float pred_ys = (pred_y1[idx] + pred_y2[idx]) / 2;
    float pred_w = pred_x2[idx] - pred_x1[idx];
    float pred_h = pred_y2[idx] - pred_y1[idx];

    float diff_xs = (pred_xs - instance_size / 2) / scale_z;
    float diff_ys = (pred_ys - instance_size / 2) / scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    float lr_ = penalty[idx] * cls_score_sigmoid[idx] * lr;

    float res_xs = float(target_pos.x + diff_xs);
    float res_ys = float(target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr_) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr_) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr_) + lr_ * res_w;
    target_sz.y = target_sz.y * (1 - lr_) + lr_ * res_h;
}

void LightTrack::track(const uint8_t *img) {
    float hc_z = target_sz.y + context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);
    float scale_z = exemplar_size / s_z;

    float d_search = (instance_size - exemplar_size) / 2;
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;

    cv::Mat img_(ori_img_h, ori_img_w, CV_8UC3, (void*)img, ori_img_w * 3);
    cv::Mat x_crop = get_subwindow_tracking(img_, target_pos, instance_size, int(s_x));

    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    update(x_crop, scale_z);

    target_pos.x = std::max(0, std::min(ori_img_w, target_pos.x));
    target_pos.y = std::max(0, std::min(ori_img_h, target_pos.y));
    target_sz.x = float(std::max(10, std::min(ori_img_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, std::min(ori_img_h, int(target_sz.y))));
}

void LightTrack::grids() {
    int sz = score_size;
    grid_to_search_x.resize(sz * sz, 0);
    grid_to_search_y.resize(sz * sz, 0);
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            grid_to_search_x[i * sz + j] = j * total_stride;
            grid_to_search_y[i * sz + j] = i * total_stride;
        }
    }
}

cv::Mat LightTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz) {
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad   = int(std::max(0, -context_xmin));
    int top_pad    = int(std::max(0, -context_ymin));
    int right_pad  = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat im_path_original;
    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0) {
        cv::Mat te_im;
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0.f);
        im_path_original = te_im(
            cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    } else {
        im_path_original = im(
            cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));
    return im_path;
}
