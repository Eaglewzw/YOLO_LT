#include "net.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>

class Noop : public ncnn::Layer
{
};
DEFINE_LAYER_CREATOR(Noop)


/**
 *  "0": "brid",
 *  "1": "car",
 *  "2": "drone",
 *  "3": "person"
 * 
 */
static const char* class_names[] = {"brid","car", "drone", "person"};

void softmax(std::vector<float>& array) 
{
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < array.size(); i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < array.size(); i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < array.size(); i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < array.size(); i++) {
        array[i] /= sum;
    }
}




static int detect_mobilenetv2(const cv::Mat& bgr, std::vector<float>& objects)
{
    ncnn::Net mobilenetv2;

    mobilenetv2.opt.use_vulkan_compute = true;

    mobilenetv2.register_custom_layer("Silence", Noop_layer_creator);

    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    if (mobilenetv2.load_param("/home/verse/C_plus_plus/Graduation_Code/Light_DT/03_MobileNetV2/model/MobileNetV2_K.ncnn.param"))
        exit(-1);
    if (mobilenetv2.load_model("/home/verse/C_plus_plus/Graduation_Code/Light_DT/03_MobileNetV2/model/MobileNetV2_K.ncnn.bin"))
        exit(-1);

    const int target_size = 224;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);


    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);




    ncnn::Extractor ex = mobilenetv2.create_extractor();


    // ex.input("input_1", in);
    ex.input("in0", in);

    ncnn::Mat out;

    // ex.extract("output_1", out);
    ex.extract("out0", out);


    printf("%d %d %d\n", out.w, out.h, out.c);
    float *cls_score_data = (float *) out.data;

    float prob[4];
    


    objects.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        objects[j] = out[j];
        prob[j] = out[j]; 
        printf("cls_scores[%d]=%f\n", j, objects[j]);
    }

    return 0;
}



int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> objects;
    detect_mobilenetv2(m, objects);
    softmax(objects);
    for (int i = 0; i < objects.size(); i++) {
        std::cout << class_names[i] << ":" << objects[i] << std::endl;
    }

    return 0;
}