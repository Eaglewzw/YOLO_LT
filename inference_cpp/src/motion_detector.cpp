#include "motion_detector.h"
#include <numeric>
#include <iostream>
#include <cmath>

MotionDetector::MotionDetector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "MotionDetector")
    , session_(nullptr)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), opts);
    std::cout << "MotionDetector classifier loaded: " << model_path << std::endl;
}

// ==================== Motion Compensation ====================

MotionDetector::CompensateResult MotionDetector::motionCompensate(
    const cv::Mat& gray1, const cv::Mat& gray2)
{
    int width = gray2.cols, height = gray2.rows;
    int gridSizeW = 32 * 3, gridSizeH = 24 * 3;

    std::vector<cv::Point2f> pts_prev;
    int grid_numW = width / gridSizeW - 1;
    int grid_numH = height / gridSizeH - 1;
    for (int i = 0; i < grid_numW; i++)
        for (int j = 0; j < grid_numH; j++)
            pts_prev.emplace_back(i * gridSizeW + gridSizeW / 2.0f,
                                  j * gridSizeH + gridSizeH / 2.0f);

    std::vector<cv::Point2f> pts_cur;
    std::vector<uchar> st;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(gray1, gray2, pts_prev, pts_cur, st, err,
        cv::Size(15, 15), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    std::vector<cv::Point2f> good_new, good_old;
    std::vector<double> motion_dist;
    for (size_t i = 0; i < st.size(); i++) {
        if (!st[i]) continue;
        double dx = pts_cur[i].x - pts_prev[i].x;
        double dy = pts_cur[i].y - pts_prev[i].y;
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist > 50) continue;
        good_new.push_back(pts_cur[i]);
        good_old.push_back(pts_prev[i]);
        motion_dist.push_back(dist);
    }

    double avg_dist = motion_dist.empty() ? 0 :
        std::accumulate(motion_dist.begin(), motion_dist.end(), 0.0) / motion_dist.size();

    cv::Mat H;
    cv::Mat identity = (cv::Mat_<double>(3, 3) << 0.999, 0, 0, 0, 0.999, 0, 0, 0, 1);
    if (good_old.size() < 9) {
        H = identity;
    } else {
        H = cv::findHomography(good_new, good_old, cv::RANSAC, 3.0);
        if (H.empty()) H = identity;
    }

    cv::Mat compensated;
    cv::warpPerspective(gray1, compensated, H, cv::Size(width, height),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

    // Compute edge mask
    std::vector<cv::Point2f> vertex = {
        {0, 0}, {(float)width, 0}, {(float)width, (float)height}, {0, (float)height}};
    cv::Mat H_inv = H.inv();
    std::vector<cv::Point2f> vertex_trans;
    cv::perspectiveTransform(vertex, vertex_trans, H_inv);

    cv::Mat mask_img = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    std::vector<cv::Point> poly;
    for (auto& p : vertex_trans) poly.emplace_back(int(p.x), int(p.y));
    std::vector<std::vector<cv::Point>> polys = {poly};
    cv::fillPoly(mask_img, polys, 255);
    mask_img = 255 - mask_img;

    return {compensated, mask_img, avg_dist};
}

// ==================== Classifier (ONNX) ====================

int MotionDetector::classify(const cv::Mat& crop) {
    cv::Mat resized, rgb;
    cv::resize(crop, resized, cv::Size(CLASSIFY_SIZE, CLASSIFY_SIZE));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    std::vector<float> input_data(3 * CLASSIFY_SIZE * CLASSIFY_SIZE);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; c++)
        memcpy(input_data.data() + c * CLASSIFY_SIZE * CLASSIFY_SIZE,
               channels[c].data, CLASSIFY_SIZE * CLASSIFY_SIZE * sizeof(float));

    std::array<int64_t, 4> shape = {1, 3, CLASSIFY_SIZE, CLASSIFY_SIZE};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem, input_data.data(), input_data.size(), shape.data(), shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
        input_names, &tensor, 1, output_names, 1);

    const float* out = outputs[0].GetTensorData<float>();
    // out[0] = non-drone, out[1] = drone — return argmax
    return (out[1] > out[0]) ? 1 : 0;
}

// ==================== Utility ====================

cv::Rect MotionDetector::enlargeBox(int x, int y, int w, int h,
                                     int pad, int img_w, int img_h) {
    int x1 = std::max(0, x - pad);
    int y1 = std::max(0, y - pad);
    int w1 = std::min(w + pad * 2, img_w - x1 - 1);
    int h1 = std::min(h + pad * 2, img_h - y1 - 1);
    return cv::Rect(x1, y1, w1, h1);
}

// ==================== Main Detection Pipeline ====================

cv::Rect MotionDetector::detect(const cv::Mat& prevFrame, const cv::Mat& currentFrame) {
    int width = currentFrame.cols, height = currentFrame.rows;

    // 1. Gaussian blur + grayscale
    cv::Mat prev_blur, curr_blur, prev_gray, curr_gray;
    cv::GaussianBlur(prevFrame, prev_blur, cv::Size(11, 11), 0);
    cv::GaussianBlur(currentFrame, curr_blur, cv::Size(11, 11), 0);
    cv::cvtColor(prev_blur, prev_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_blur, curr_gray, cv::COLOR_BGR2GRAY);

    // 2. Motion compensation
    auto [compensated, mask, avg_dist] = motionCompensate(prev_gray, curr_gray);

    // 3. Adaptive frame differencing
    cv::Mat frameDiff;
    cv::absdiff(curr_gray, compensated, frameDiff);
    double fix_coef = cv::mean(frameDiff)[0];
    int threshold_val = 5 + int(fix_coef);

    cv::Mat thresh;
    cv::threshold(frameDiff, thresh, threshold_val, 255, cv::THRESH_BINARY);

    // 4. Subtract edge mask + median blur
    thresh = thresh - mask;
    cv::medianBlur(thresh, thresh, 5);

    // 5. Morphology: opening + closing
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);

    // 6. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Filter by area and aspect ratio
    struct Candidate { cv::Rect rect; };
    std::vector<Candidate> candidates;
    for (const auto& contour : contours) {
        cv::Rect r = cv::boundingRect(contour);
        double area = cv::contourArea(contour);
        double ratio = (double)r.width / r.height;
        if (area > 16 && area < 3000 && ratio > 0.6 && ratio < 3.0) {
            candidates.push_back({r});
        }
    }

    if (candidates.size() > 50) return {};

    // 7. Local optical flow verification + classification
    cv::TermCriteria lk_criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.03);

    for (auto& cand : candidates) {
        // Enlarge box
        cv::Rect box = enlargeBox(cand.rect.x, cand.rect.y,
            cand.rect.width, cand.rect.height, 2, width, height);

        double ratio = (double)box.width / box.height;
        if (ratio < 0.6 || ratio > 3.0) continue;
        if (box.width < 4 || box.height < 4) continue;

        cv::Mat crop_curr = curr_gray(box);
        cv::Mat crop_comp = compensated(box);

        // Shi-Tomasi corner detection
        std::vector<cv::Point2f> pts1;
        cv::goodFeaturesToTrack(crop_curr, pts1, 30, 0.15, 3, cv::noArray(), 3);
        if (pts1.empty()) continue;

        // Local optical flow
        std::vector<cv::Point2f> pts2;
        std::vector<uchar> st;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(crop_comp, crop_curr, pts1, pts2, st, err,
            cv::Size(15, 15), 3, lk_criteria);

        std::vector<double> local_dist, local_theta;
        for (size_t i = 0; i < st.size(); i++) {
            if (!st[i]) continue;
            double dx = pts2[i].x - pts1[i].x;
            double dy = pts2[i].y - pts1[i].y;
            local_dist.push_back(std::sqrt(dx * dx + dy * dy));
            local_theta.push_back(57.3 * std::atan2(-dy, -dx));
        }

        if (local_dist.empty()) continue;

        // Motion statistics filtering
        double mean_dist = std::accumulate(local_dist.begin(), local_dist.end(), 0.0) / local_dist.size();
        if (mean_dist < 1.0) continue;

        double mean_theta = std::accumulate(local_theta.begin(), local_theta.end(), 0.0) / local_theta.size();
        double sq_sum_theta = 0, sq_sum_dist = 0;
        for (size_t i = 0; i < local_dist.size(); i++) {
            sq_sum_theta += (local_theta[i] - mean_theta) * (local_theta[i] - mean_theta);
            sq_sum_dist += (local_dist[i] - mean_dist) * (local_dist[i] - mean_dist);
        }
        double std_theta = std::sqrt(sq_sum_theta / local_theta.size());
        double std_dist  = std::sqrt(sq_sum_dist / local_dist.size());

        if (std::abs(mean_theta) > 1e-6 && std_theta / std::abs(mean_theta) > 0.8) continue;
        if (mean_dist > 1e-6 && std_dist / mean_dist > 0.8) continue;

        // 8. Classify with ONNX model
        cv::Mat color_crop = currentFrame(box);
        int cls = classify(color_crop);
        if (cls == 1) {
            return box;
        }
    }

    return {};
}
