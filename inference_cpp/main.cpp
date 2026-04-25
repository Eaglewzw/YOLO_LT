#include "config.h"
#include <filesystem>

// ==================== Configuration ====================
constexpr int    VISUAL_FAIL_THRESHOLD = 30;
constexpr int    MOD_FAIL_THRESHOLD    = 10;
constexpr int    YOLO_VERIFY_INTERVAL  = 30;
constexpr float  YOLO_VERIFY_CONF      = 0.45f;
constexpr float  TRACK_SCORE_THRESHOLD = 0.98f;
constexpr float  ROI_EXPANSION_FACTOR  = 3.0f;

struct EnableConfig {
    bool visual_detect = true;
    bool motion_detect = true;
    bool tracking      = true;
};

// ==================== Utility Functions ====================

cv::Rect safeRect(const cv::Rect& r, const cv::Size& imgSize) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width,  imgSize.width  - x);
    int h = std::min(r.height, imgSize.height - y);
    return cv::Rect(x, y, std::max(0, w), std::max(0, h));
}

void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect& rect) {
    rect.x = std::max(0, pos.x - int(sz.x / 2));
    rect.y = std::max(0, pos.y - int(sz.y / 2));
    rect.width  = int(sz.x);
    rect.height = int(sz.y);
}

cv::Mat getSafeROI(const cv::Mat& frame, const cv::Rect& bbox, float expansion) {
    int img_h = frame.rows, img_w = frame.cols;
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    float new_w = bbox.width * expansion;
    float new_h = bbox.height * expansion;

    int x1 = std::max(0, int(cx - new_w / 2));
    int y1 = std::max(0, int(cy - new_h / 2));
    int x2 = std::min(img_w, int(cx + new_w / 2));
    int y2 = std::min(img_h, int(cy + new_h / 2));

    if (x2 - x1 < 10 || y2 - y1 < 10) return {};
    return frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
}

bool initTracker(LightTrack& tracker, const cv::Mat& frame, const cv::Rect& bbox) {
    Bbox box;
    box.x0 = bbox.x;
    box.y0 = bbox.y;
    box.x1 = bbox.x + bbox.width;
    box.y1 = bbox.y + bbox.height;

    float cx = (box.x0 + box.x1) / 2.0f;
    float cy = (box.y0 + box.y1) / 2.0f;
    if (cx >= frame.cols || cy >= frame.rows ||
        box.x0 >= frame.cols || box.y0 >= frame.rows) {
        return false;
    }

    tracker.init(frame.data, box, frame.rows, frame.cols);
    return true;
}

// ==================== Main ====================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./Light_DT <model.onnx> <video_path>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];

    namespace fs = std::filesystem;
    fs::path project_dir = fs::canonical("/proc/self/exe").parent_path().parent_path();
    std::string model_dir = (project_dir / "model").string();

    EnableConfig config;

    // ==================== Model Initialization ====================

    // YOLO detector (ONNX Runtime)
    YoloDetector yolo(model_path);

    // LightTrack tracker (ONNX Runtime)
    std::string init_model  = model_dir + "/lighttrack_init";
    std::string update_model = model_dir + "/lighttrack_update";
    auto tracker = std::make_unique<LightTrack>(init_model, update_model);

    // MOD motion detector (ONNX Runtime)
    std::unique_ptr<MotionDetector> mod_detector;
    if (config.motion_detect) {
        std::string classifier_path = model_dir + "/Net_best.onnx";
        mod_detector = std::make_unique<MotionDetector>(classifier_path);
    }

    // ==================== Video Setup ====================

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }

    cv::Mat prevFrame;
    if (!cap.read(prevFrame)) {
        std::cerr << "Cannot read video file" << std::endl;
        return -1;
    }

    int frame_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_rate   = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (frame_rate <= 0) frame_rate = 60;

    cv::VideoWriter outputVideo;
    std::string output_path = (project_dir / "output_video.mp4").string();
    outputVideo.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                     frame_rate, cv::Size(frame_width, frame_height), true);
    if (!outputVideo.isOpened()) {
        std::cerr << "Error: Unable to open output video file!" << std::endl;
        return -1;
    }

    cv::namedWindow("demo", cv::WINDOW_NORMAL);
    cv::resizeWindow("demo", 640, 640);

    // ==================== State ====================

    bool tracking_state = false;
    int  frame_count    = 0;
    int  visual_fail_count = 0;
    int  mod_fail_count    = 0;
    double current_fps     = 0.0;

    std::cout << "System started. Config: V=" << config.visual_detect
              << " M=" << config.motion_detect
              << " T=" << config.tracking << std::endl;

    // ==================== Main Loop ====================

    cv::Mat currentFrame;
    while (cap.read(currentFrame)) {
        if (currentFrame.empty()) break;

        auto t_start = std::chrono::high_resolution_clock::now();
        frame_count++;

        cv::Mat displayFrame = currentFrame.clone();

        // ==================== Branch 1: Search Mode ====================
        if (!tracking_state) {
            cv::Rect detected_bbox;
            bool detected = false;
            std::string search_mode;

            // Strategy A: Visual Detection (YOLO)
            bool can_use_visual = config.visual_detect;
            bool force_visual = can_use_visual && !config.motion_detect;

            if (can_use_visual && (visual_fail_count < VISUAL_FAIL_THRESHOLD || force_visual)) {
                search_mode = "VISUAL";
                auto detections = yolo.detect(currentFrame, CONF_THRESH, NMS_THRESH);

                float best_score = 0;
                for (const auto& det : detections) {
                    if (det.conf > best_score) {
                        best_score = det.conf;
                        detected_bbox = yolo.get_rect(currentFrame, det.bbox);
                    }
                }

                if (best_score > CONF_THRESH) {
                    detected = true;
                    visual_fail_count = 0;
                    mod_fail_count = 0;
                } else {
                    visual_fail_count++;
                }
            }
            // Strategy B: Motion Detection (MOD)
            else if (config.motion_detect && mod_detector) {
                search_mode = "MOTION";
                cv::Rect mod_rect = mod_detector->detect(prevFrame, currentFrame);

                if (mod_rect.width > 0 && mod_rect.height > 0) {
                    detected_bbox = mod_rect;
                    detected = true;
                    mod_fail_count = 0;
                } else {
                    mod_fail_count++;
                    if (mod_fail_count >= MOD_FAIL_THRESHOLD) {
                        std::cout << "Frame " << frame_count
                                  << ": MOD failed " << MOD_FAIL_THRESHOLD
                                  << " frames -> reset to YOLO" << std::endl;
                        visual_fail_count = 0;
                        mod_fail_count = 0;
                    }
                }
            }

            // State transition: init tracker
            if (detected && config.tracking) {
                if (initTracker(*tracker, currentFrame, detected_bbox)) {
                    tracking_state = true;
                    visual_fail_count = 0;
                    mod_fail_count = 0;
                    std::cout << "Frame " << frame_count
                              << ": Target locked (" << search_mode << ") -> tracking" << std::endl;
                }
            } else {
                std::string fail_info = (search_mode == "VISUAL")
                    ? std::to_string(visual_fail_count) + "/" + std::to_string(VISUAL_FAIL_THRESHOLD)
                    : std::to_string(mod_fail_count) + "/" + std::to_string(MOD_FAIL_THRESHOLD);
                cv::Scalar status_color = (search_mode == "VISUAL")
                    ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255);
                cv::putText(displayFrame, "SEARCHING (" + search_mode + " " + fail_info + ")",
                    cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);
            }
        }
        // ==================== Branch 2: Tracking Mode ====================
        else {
            if (tracker->target_pos.x >= currentFrame.cols ||
                tracker->target_pos.y >= currentFrame.rows) {
                tracking_state = false;
                continue;
            }

            tracker->track(currentFrame.data);
            float score = tracker->track_score;

            cv::Rect rect;
            cxy_wh_2_rect(tracker->target_pos, tracker->target_sz, rect);
            rect = safeRect(rect, currentFrame.size());

            // Periodic YOLO verification
            if (config.visual_detect && frame_count % YOLO_VERIFY_INTERVAL == 0 &&
                rect.width > 0 && rect.height > 0) {

                cv::Mat roi = getSafeROI(currentFrame, rect, ROI_EXPANSION_FACTOR);
                if (!roi.empty()) {
                    auto roi_dets = yolo.detect(roi, CONF_THRESH, NMS_THRESH);
                    float best_verify = 0;
                    for (const auto& d : roi_dets) {
                        best_verify = std::max(best_verify, d.conf);
                    }

                    if (best_verify < YOLO_VERIFY_CONF) {
                        std::cout << "Frame " << frame_count
                                  << ": YOLO verify failed (" << best_verify
                                  << " < " << YOLO_VERIFY_CONF << ") -> search" << std::endl;
                        tracking_state = false;
                        score = 0;
                    }
                }
            }

            // Score-based track loss
            if (!tracking_state || score <= TRACK_SCORE_THRESHOLD) {
                if (tracking_state) {
                    std::cout << "Frame " << frame_count
                              << ": Track lost (score=" << score << ") -> search" << std::endl;
                }
                tracking_state = false;
                visual_fail_count = 0;
                mod_fail_count = 0;
            } else {
                cv::rectangle(displayFrame, rect, cv::Scalar(0, 255, 255), 2);
                cv::putText(displayFrame, "Track: " + std::to_string(score).substr(0, 4),
                    cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);
            }
        }

        // ==================== Display ====================
        prevFrame = currentFrame.clone();

        auto t_end = std::chrono::high_resolution_clock::now();
        double instant_fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
        current_fps = 0.1 * instant_fps + 0.9 * current_fps;

        std::string state_text = tracking_state ? "TRACKING" : "SEARCHING";
        cv::Scalar state_color = tracking_state ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
        cv::putText(displayFrame, "Mode: " + state_text,
            cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2);
        cv::putText(displayFrame, "FPS: " + std::to_string(int(current_fps)),
            cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("demo", displayFrame);
        outputVideo.write(displayFrame);
        if (cv::waitKey(1) == 'q') break;
    }

    cv::destroyWindow("demo");
    cap.release();
    outputVideo.release();

    std::cout << "Process finished." << std::endl;
    return 0;
}
