#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"


#include <thread>


/***************************OpenCV库***************************/
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video.hpp"
#include "opencv2/opencv.hpp"

#include "LightTrack.h"
#include "MotionDetector.hpp"


#define USE_FP16                                      // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0                                      // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000       // ensure it exceed the maximum size in the input images !



#endif