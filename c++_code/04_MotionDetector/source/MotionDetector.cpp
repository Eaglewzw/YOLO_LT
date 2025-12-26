#include "MotionDetector.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "net.h"
#include <numeric> 

/**
 *  "0": "brid",
 *  "1": "car",
 *  "2": "drone",
 *  "3": "person"
 * 
 */
static const char* class_names[] = {"brid", "car", "drone", "person", "umbrella"};

void softmax(float* array, int size) 
{
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}




float MotionDetector::MobileNetV2_Infer(const cv::Mat& bgr)
{
    const int target_size = 60;
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);



    in.substract_mean_normalize(mean_vals, norm_vals);


    ncnn::Extractor ex = MobileNetV2.create_extractor();

    ncnn::Mat out;
    ex.input("in0", in);
    ex.extract("out0", out);

    float prob[out.w];
    
    for (int j = 0; j < out.w; j++)
        prob[j] = out[j]; 

    softmax(prob, out.w);
    // std::cout << "==========================" << std::endl;
    // std::cout << std::endl;
    // for (int j = 0; j < out.w; j++)
    //     printf("%s: %.2f\n", class_names[j], prob[j]);
    // std::cout << "==========================" << std::endl;
    // std::cout << std::endl;


    return prob[2];
}




// motion_compensate函数
std::tuple<cv::Mat, cv::Mat, double> motion_compensate(const cv::Mat &frame1, const cv::Mat &frame2) 
{
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::Size winSize(15, 15);

    int width = frame2.cols;
    int height = frame2.rows;
    int gridSizeW = 32 * 3;
    int gridSizeH = 24 * 3;
    std::vector<cv::Point2f> p1;

    int grid_numW = static_cast<int>(width / gridSizeW - 1);
    int grid_numH = static_cast<int>(height / gridSizeH - 1);
    for (int i = 0; i < grid_numW; ++i) {
        for (int j = 0; j < grid_numH; ++j) {
            p1.emplace_back(static_cast<float>(i * gridSizeW + gridSizeW / 2.0), static_cast<float>(j * gridSizeH + gridSizeH / 2.0));
        }
    }

    std::vector<cv::Point2f> pts_prev(p1.begin(), p1.end());
    std::vector<cv::Point2f> pts_cur;
    std::vector<uchar> st;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(frame1, frame2, pts_prev, pts_cur, st, err, winSize, 3, termcrit);

    std::vector<cv::Point2f> good_new, good_old;
    std::vector<double> motion_distance;

    for (size_t i = 0; i < st.size(); ++i) {
        if (st[i]) {
            good_new.push_back(pts_cur[i]);
            good_old.push_back(pts_prev[i]);
            double a = pts_cur[i].x, b = pts_cur[i].y;
            double c = pts_prev[i].x, d = pts_prev[i].y;
            double dist = std::sqrt((a - c) * (a - c) + (b - d) * (b - d));
            if (dist > 50) continue;
            motion_distance.push_back(dist);
        }
    }

    double avg_dist = std::accumulate(motion_distance.begin(), motion_distance.end(), 0.0) / motion_distance.size();

    cv::Mat homography_matrix;
    if (good_old.size() < 9) {
        homography_matrix = (cv::Mat_<double>(3, 3) << 0.999, 0, 0, 0, 0.999, 0, 0, 0, 1);
    } else {
        homography_matrix = cv::findHomography(good_new, good_old, cv::RANSAC, 3.0);
    }

    cv::Mat compensated;
    cv::warpPerspective(frame1, compensated, homography_matrix, cv::Size(width, height), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

    std::vector<cv::Point2f> vertex = {cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(width, height), cv::Point2f(0, height)};
    cv::Mat homo_inv = homography_matrix.inv();
    std::vector<cv::Point2f> vertex_trans;
    cv::perspectiveTransform(vertex, vertex_trans, homo_inv);

    cv::Mat mask = cv::Mat::zeros(frame1.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> fillContAll = {std::vector<cv::Point>(vertex_trans.begin(), vertex_trans.end())};
    cv::fillPoly(mask, fillContAll, 255);
    mask = 255 - mask;

    return std::make_tuple(compensated, mask, avg_dist);
}






cv::Rect MotionDetector::MotionDiff(cv::Mat prevFrame, cv::Mat currentFrame)
{
    double avg_dist;
    cv::Rect resultRect;

    cv::Mat img_compensate, mask;

    cv::Mat resultFrame = currentFrame.clone();

    /******************1.将background和frame转为灰度图******************/
    /***为了提高计算机的运算速度，图像处理前一般将图像转成灰度图***/
    cv::Mat prevFrameGray, currentFrameGray;
    cvtColor(prevFrame,     prevFrameGray,      cv::COLOR_BGR2GRAY);
    cvtColor(currentFrame,  currentFrameGray,   cv::COLOR_BGR2GRAY);



    std::tuple<cv::Mat, cv::Mat, double> motion_compensate_tmp;
    motion_compensate_tmp =  motion_compensate(prevFrameGray, currentFrameGray);
   
    std::tie(img_compensate, mask, avg_dist) = motion_compensate_tmp; // 提取 tuple 中的三个数据



    /******************2.将background和frame做差******************/
    cv::Mat frameDiff;
    absdiff(currentFrameGray, img_compensate, frameDiff);
    // absdiff(prevFrameGray, currentFrameGray, frameDiff);
//    imshow("frameDiff", frameDiff);


    /********************3.对差值图frameDiff进行阈值化处理********************/
    /***********图像二值化***********/
    cv::Mat diffThresh;
    threshold(frameDiff, frameDiff, 25, 255, cv::THRESH_BINARY);
    // imshow("diffThresh", diffThresh);


    /****************************4.腐蚀****************************/
    /** 去掉白色噪点 ****/
    cv::Mat kernel_erode = getStructuringElement(cv::MORPH_RECT,  cv::Size(3, 3));
    cv::Mat kernel_dilate = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    erode(frameDiff, frameDiff, kernel_erode);  // 对差值图进行腐蚀
//    imshow("erode", frameDiff);

    /****************************5.膨胀****************************/
    dilate(frameDiff, frameDiff, kernel_dilate);  // 对腐蚀后的图像进行膨胀
//    imshow("dilate", frameDiff);


    /*************************6.查找轮廓并绘制轮廓*************************/
    std::vector<std::vector<cv::Point>> contours; //保存关键点
    findContours(frameDiff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());


    /*********************************7.查找正外接矩形*********************************/
    int x, y, w, h;
    int num = contours.size();
    
    float max_prob = 0.6;
    for(int i = 0; i < num; i++)
    {
        approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(cv::Mat(contours_poly[i]));

        double area = cv::contourArea(contours[i]);
        int w1 = boundRect[i].width;
        int h1 = boundRect[i].height;
        
        // 计算长宽比
        double ratio_dist_1 = static_cast<double>(w1) / h1;
        double ratio_dist_2 = static_cast<double>(h1) / w1;
        if(area > 30 && area < 3000 && ratio_dist_1 <= 3 && ratio_dist_2 <= 3)
        {
            // 使用 Shi-Tomasi 角点检测器检测角点
            std::vector<cv::Point2f> corners;
            cv::Mat GLAD_crop = currentFrame(boundRect[i]);
            float mindex = MobileNetV2_Infer(GLAD_crop);  

            // std::string numStr = std::to_string(i++) + ".png";
            // cv::imwrite(numStr, GLAD_crop);
            if(mindex >= max_prob) {
                x = boundRect[i].x;
                y = boundRect[i].y;
                w = boundRect[i].width;
                h = boundRect[i].height;
                max_prob = mindex;
            } 
        }
    }



    //绘制
    if(x != 0 && y != 0 && w != 0 && h != 0){
        resultRect.x = x;
        resultRect.y = y;
        resultRect.width  = w;
        resultRect.height = h;
    }
  

    return resultRect;
}


void MotionDetector::ProcessFrameWithDiff(const cv::Mat& prevFrame,
                                          cv::Mat& currentFrame,
                                          float threshold)
{
    // 1. 用差分 + 光流补偿找运动候选框
    cv::Rect bbox = MotionDiff(prevFrame, currentFrame);

    // 2. 限制 bbox 在图像范围内
    bbox.x = std::max(0, bbox.x);
    bbox.y = std::max(0, bbox.y);
    bbox.width  = std::max(0, std::min(bbox.width,  currentFrame.cols - bbox.x));
    bbox.height = std::max(0, std::min(bbox.height, currentFrame.rows - bbox.y));

    // 3. 如果找到候选框，则分类
    if (bbox.width > 0 && bbox.height > 0) {
        cv::Mat roi = currentFrame(bbox).clone();
        float drone_prob = MobileNetV2_Infer(roi);

        // 4. 只画出大于阈值的框
        if (drone_prob > threshold) {
            std::string label = "drone_prob: " + cv::format("%.2f", drone_prob);
            cv::rectangle(currentFrame, bbox, cv::Scalar(0, 255, 0), 2);
            cv::putText(currentFrame, label, bbox.tl(),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}



class Noop : public ncnn::Layer
{
};
DEFINE_LAYER_CREATOR(Noop)

MotionDetector::MotionDetector(/* args */)
{

    MobileNetV2.opt.use_vulkan_compute = true;

    MobileNetV2.register_custom_layer("Silence", Noop_layer_creator);

    if (MobileNetV2.load_param("/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/model/MobileNetV2/MobileNetV2_K.ncnn.param"))
        exit(-1);
    if (MobileNetV2.load_model("/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/model/MobileNetV2/MobileNetV2_K.ncnn.bin"))
        exit(-1);


}

MotionDetector::~MotionDetector()
{


}