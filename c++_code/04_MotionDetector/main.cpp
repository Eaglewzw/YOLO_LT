#include <opencv2/opencv.hpp>
#include "MotionDetector.hpp"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " video_path" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video " << video_path << std::endl;
        return -1;
    }

    MotionDetector detector;
    cv::Mat prevFrame, currentFrame;
    bool firstFrame = true;

    double tickFreq = cv::getTickFrequency();
    double startTick, fps = 0.0;

    // 创建可调整大小窗口并启动全屏
    cv::namedWindow("Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Detection", 640, 640); 

    while (true) {
        startTick = cv::getTickCount(); // 记录开始时间

        cap >> currentFrame;
        if (currentFrame.empty()) break;

        if (!firstFrame) {
            detector.ProcessFrameWithDiff(prevFrame, currentFrame, 0.95f);
        } else {
            firstFrame = false;
        }

        // 计算 FPS
        double elapsed = (cv::getTickCount() - startTick) / tickFreq;
        fps = 1.0 / elapsed;

        // 显示 FPS 到画面
        std::string fpsText = "FPS: " + std::to_string((int)fps);
        cv::putText(currentFrame, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Detection", currentFrame);
        char c = (char)cv::waitKey(1); // waitKey 改小可以提高 FPS
        if (c == 'q' || c == 27) break;

        prevFrame = currentFrame.clone();
    }
    return 0;
}
