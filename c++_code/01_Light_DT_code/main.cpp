#include "main.h"




/**************************************************************************************/

//全局变量
int glad_local_flag = 0;
int init_trackwindow_flag = 0;

int glad_appearance_detect_flag = 1;
int glad_motion_detect_flag = 0;

/**************************************************************************************/


// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}



void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}




void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect &rect)
{
    rect.x = max(0, pos.x - int(sz.x / 2));
    rect.y = max(0, pos.y - int(sz.y / 2));
    rect.width = int(sz.x);
    rect.height = int(sz.y);
}


double compareHist(cv::Mat src_origin_1, cv::Mat src_origin_2)
{
    // 转换到 HSV , 图片是RGB格式用CV_RGB2HSV
    cv::Mat src_1, src_2;
    cv::cvtColor( src_origin_1 , src_1 , cv::COLOR_BGR2HSV );
    cv::cvtColor( src_origin_2, src_2, cv::COLOR_BGR2HSV );

    // 对hue通道使用30个bin,对saturatoin通道使用32个bin
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue的取值范围从0到256, saturation取值范围从0到180
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };

    const float* ranges[] = { h_ranges, s_ranges };
    // 使用第0和第1通道
    int channels[] = { 0, 1 };

    // 直方图
    cv::MatND src_1_hist,src_2_hist;
    // 计算HSV图像的直方图
    cv::calcHist( &src_1 , 1, channels, Mat(), src_1_hist, 2, histSize, ranges, true, false );
    cv::normalize( src_1_hist, src_1_hist, 0, 1, cv::NORM_MINMAX, -1, Mat() );
    cv::calcHist( &src_2 , 1, channels, Mat(), src_2_hist, 2, histSize, ranges, true, false );
    cv::normalize( src_2_hist, src_2_hist, 0, 1, cv::NORM_MINMAX, -1, Mat() );

    //对比方法
    double result = cv::compareHist( src_1_hist, src_2_hist, 0 );
    return result;
}


double ssimDetect(cv::Mat imgOrg, cv::Mat imgComp)
{
    double C1 = 6.5025, C2 = 58.5225;
    
    // cv::Mat image_ref = cv::imread(imgOrg, CV_LOAD_IMAGE_GRAYSCALE);
    // cv::Mat image_obj = cv::imread(imgComp, CV_LOAD_IMAGE_GRAYSCALE);
     cv::Mat image_ref, image_obj;

    cvtColor(imgOrg, image_ref, COLOR_BGR2GRAY);

    cvtColor(imgComp, image_obj, COLOR_BGR2GRAY);

    int width = image_ref.cols;
    int height = image_ref.rows;
    int width2 = image_obj.cols;
    int height2 = image_obj.rows;
    double mean_x = 0;
    double mean_y = 0;
    double sigma_x = 0;
    double sigma_y = 0;
    double sigma_xy = 0;
    for (int v = 0; v < height; v++)
    {
        for (int u = 0; u < width; u++)
        {
            mean_x += image_ref.at<uchar>(v, u);
            mean_y += image_obj.at<uchar>(v, u);

        }
    }
    mean_x = mean_x / width / height;
    mean_y = mean_y / width / height;
    for (int v = 0; v < height; v++)
    {
        for (int u = 0; u < width; u++)
        {
            sigma_x += (image_ref.at<uchar>(v, u) - mean_x)* (image_ref.at<uchar>(v, u) - mean_x);
            sigma_y += (image_obj.at<uchar>(v, u) - mean_y)* (image_obj.at<uchar>(v, u) - mean_y);
            sigma_xy += abs((image_ref.at<uchar>(v, u) - mean_x)* (image_obj.at<uchar>(v, u) - mean_y));
        }
    }
    sigma_x = sigma_x / (width*height - 1);
    sigma_y = sigma_y / (width*height - 1);
    sigma_xy = sigma_xy / (width*height - 1);
    double fenzi = (2 * mean_x*mean_y + C1) * (2 * sigma_xy + C2);
    double fenmu = (mean_x*mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
    double ssim = fenzi / fenmu;
    return ssim;
}





int main(int argc, char** argv) 
{
    cudaSetDevice(DEVICE);


    int frameCount = 1;
    cv::Mat currentFrame;               /* currentFrame:当前帧 */
    cv::Mat prevFrame;                  /* prevFrame   :处理帧 */


    MotionDetector mMotionDetector;


    std::string engine_name = "/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/model/yolov5s_GLAD.engine";
    std::string img_dir = "/home/verse/Videos/phantom13.mp4";


    if (argc < 3) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 [.engine] <image_path | video_path | camera_device>  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    engine_name = std::string(argv[1]);
    img_dir = std::string(argv[2]);


    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();


    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    int fcount = 0;
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);


    /****************************************Read video****************************************/ 
    cv::VideoCapture CvCapture;
    bool ret = CvCapture.open(img_dir);
    if (!ret)
        std::cout << "Open cap failed!" << std::endl; // Exit if video not opened.

    bool ok = CvCapture.read(currentFrame);
    if (!ok)
        std::cout<< "Cannot read video file" << std::endl;

    cv::namedWindow("demo", cv::WINDOW_NORMAL); // Select a rect.
    cv::resizeWindow("demo", 640, 640); 




    /***********************************局部跟踪器初始化***********************************/ 
    cv::Rect trackWindow;
    cv::Mat init_window;
    std::string init_model = "/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/model/LightTrcak/lighttrack_init";
    std::string update_model = "/home/verse/C_plus_plus/Graduation_Code/Light_DT/01_Light_DT_code/model/LightTrcak/lighttrack_update";

   
    LightTrack *siam_tracker;
    siam_tracker = new LightTrack(init_model.c_str(), update_model.c_str());


    /***********************************保存视频***********************************/

   VideoWriter outputVideo;// 打开视频写入器
   // 检查目标文件是否存在，如果不存在则创建一个空文件
   const char* outputPath = "./output_video.mp4";
   std::ifstream ifile(outputPath);
   if (!ifile) {
       std::ofstream outfile(outputPath);
       outfile.close();
   }

    int frame_width  = static_cast<int>(CvCapture.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(CvCapture.get(CAP_PROP_FRAME_HEIGHT));
    int frame_rate   = static_cast<int>(CvCapture.get(CAP_PROP_FPS));
    int frame_ex     = static_cast<int>(CvCapture.get(CAP_PROP_FOURCC)); // Get Codec Type- Int form
    if (frame_rate <= 0) 
    {
        frame_rate = 60; // 如果无法检测到帧率，手动设置为60
    }

   outputVideo.open("./output_video.mp4",  VideoWriter::fourcc('H', '2', '6', '4'), frame_rate, Size(frame_width, frame_height), true);
   if (!outputVideo.isOpened()) 
   {
       std::cerr << "Error: Unable to open output video file!" << std::endl;
       return -1;
   }


    while(CvCapture.read(currentFrame))
    {

        if (currentFrame.empty()) {
            std::cout << "frame is empty!" << std::endl;
            break;
        }

        if(prevFrame.empty()){
            std::cout << "first frame input" << std::endl;
            prevFrame = currentFrame;
            continue;
        }


        /*************************************GLAD DETECTOR*************************************/ 
        // std::cout << "Start GLAD detector inference ..." << std::endl;
        // std::cout << "==========================" << std::endl;

        double t = (double)cv::getTickCount();     // Start timer   


        if(glad_local_flag == 0) {

            if(glad_appearance_detect_flag == 1){
                float prop_sore = 0;
                fcount = 1;
                float* buffer_idx = (float*)buffers[inputIndex];
                for (int b = 0; b < fcount; b++) {
                    imgs_buffer[b] = currentFrame;
                    size_t  size_image = currentFrame.cols * currentFrame.rows * 3;
                    size_t  size_image_dst = INPUT_H * INPUT_W * 3;
                    memcpy(img_host, currentFrame.data, size_image); //copy data to pinned memory
                    CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream)); //copy data to device memory
                    preprocess_kernel_img(img_device, currentFrame.cols, currentFrame.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
                    buffer_idx += size_image_dst;
                }
                doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
                std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
                for (int b = 0; b < fcount; b++) {
                    auto& res = batch_res[b];
                    nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
                }

                cv::Rect r;
                for (int b = 0; b < fcount; b++) {
                    auto& res = batch_res[b];
                    cv::Mat img = imgs_buffer[b];
                    for (size_t j = 0; j < res.size(); j++) {
                        float score = res[j].conf;
                        if((float)res[j].conf > prop_sore) {
                            r = get_rect(img, res[j].bbox);
                            trackWindow = r;
                            prop_sore = (float)res[j].conf;
                        }

                    }
                }
                fcount = 0;
                if(prop_sore > 0.5) {
                    cv::putText(currentFrame, "Global YOLO Detection Success", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 255), 2);
                    // cv::rectangle(currentFrame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    glad_local_flag = 1;
                    init_trackwindow_flag = 0;

                    glad_motion_detect_flag = 0;     //运动检测器关闭
                }else{
                    cv::putText(currentFrame, "Global YOLO and MOD Failed", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 255), 2);

                    glad_local_flag = 0;
                    glad_motion_detect_flag = 1;      //运动检测器开启
                }

            }
            
            /*******************************运动检测器**************************/ 
            static cv::Rect a_motionRectBoxes;
            static cv::Rect b_motionRectBoxes;

            // if(glad_motion_detect_flag == 1){
   
            //     if(frameCount == 1){
            //         mMotionDetector.MotionDiff(prevFrame, currentFrame);
            //     } else {
            //         a_motionRectBoxes = mMotionDetector.MotionDiff(prevFrame, currentFrame);
            //     }

            //     double asim = 0;
            //     if(a_motionRectBoxes.area() > 0 && b_motionRectBoxes.area() > 0)
            //         asim = ssimDetect(currentFrame(a_motionRectBoxes), currentFrame(b_motionRectBoxes));
            //     if(asim > 0.2){
            //         cv::putText(currentFrame, "GLAD MOTION DETECTOR SUCCESS", cv::Point(800, 100), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 0, 0), 2); 
            //         if(a_motionRectBoxes.area() > b_motionRectBoxes.area()){
            //             trackWindow = a_motionRectBoxes;
            //         }else{
            //             trackWindow = b_motionRectBoxes;
            //         }
             
            //         glad_local_flag = 1;
            //         init_trackwindow_flag = 0;

            //     }else{
            //         glad_local_flag = 0;
            //         glad_motion_detect_flag = 0;  
            //         // cv::putText(currentFrame, "GLAD MOTION DETECTOR FAIL", cv::Point(800, 100), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 0, 0), 2); 
            //     }

            // }
            // prevFrame = currentFrame.clone();   
        }


      

        /************************************LOACL TRACK************************************/
        if(glad_local_flag == 1) {

            /***********Initialize tracker with first frame and rect***********/ 
            Bbox box;
            uint8_t *img = currentFrame.data;
            if(init_trackwindow_flag == 0){
                std::cout << "Start track init ..." << std::endl;
                std::cout << "==========================" << std::endl;
                
                box.x0 = trackWindow.x;
                box.x1 = trackWindow.x + trackWindow.width;
                box.y0 = trackWindow.y;
                box.y1 = trackWindow.y + trackWindow.height;



                if( box.x0 >= currentFrame.cols  ||
                    ((box.x0 + (box.x1 - box.x0) / 2) >= currentFrame.cols) ||
                    ((box.y0 + (box.y1 - box.y0) / 2) >= currentFrame.rows) ||
                    box.y0 >= currentFrame.rows)
                {
                    glad_local_flag = 0;
                    continue;
                }
                    

                siam_tracker->init(img, box, currentFrame.rows, currentFrame.cols);
                std::cout << "==========================" << std::endl;
                std::cout << "Init done!" << std::endl;
                std::cout << std::endl;
               
                currentFrame(trackWindow).copyTo(init_window);
                init_trackwindow_flag = 1;
            }


            uint8_t * img_track = currentFrame.data;


            if( siam_tracker->target_pos.x >= currentFrame.cols  ||
                siam_tracker->target_pos.y >= currentFrame.rows)
            {
                glad_local_flag = 0;
                continue;
            }

            siam_tracker->track(img_track); 



            cv::Rect rect;
            cxy_wh_2_rect(siam_tracker->target_pos, siam_tracker->target_sz, rect);

            // Boundary judgment.
            cv::Mat track_window;
            if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= currentFrame.cols && 0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= currentFrame.rows)
            {
                currentFrame(rect).copyTo(track_window);
                // 对比初始框和跟踪框的相似度，从而判断是否跟丢（因为LighTrack的得分输出不具有判别性，所以通过后处理引入判断跟丢机制）
                // double score = compareHist(init_window, track_window);
                double score = ssimDetect(init_window, track_window);
                std::cout << "Similarity score: " << score << std::endl;

                // // // 显示初始框和跟踪框
                // cv::imshow("init_window", init_window);
                // cv::waitKey(10);
                // cv::imshow("track_window", track_window);
                // cv::waitKey(10);

                // 相似度大于0.5的情况才进行矩形框标注
//                if (score >= 0.5) {
//                    cv::putText(currentFrame, "LOCAL TRACK SUCCESS", cv::Point(800, 60), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2);
//                    cv::rectangle(currentFrame, rect, cv::Scalar(0, 0, 255), 2);
//                } else {
//                    cv::putText(currentFrame, "GLAD TRACK FAIL", cv::Point(800, 60), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2);
//                    glad_local_flag = 0;
//                    init_trackwindow_flag = 0;
//                }
                if(siam_tracker->target_pos_change() == 0) {
                    cv::putText(currentFrame, "Local Track Success", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
                    cv::rectangle(currentFrame, rect, cv::Scalar(0, 0, 255), 1);
                } else {
                    cv::putText(currentFrame, "Local Track Failed", cv::Point(500, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
                    glad_local_flag = 0;
                    init_trackwindow_flag = 0;


                }

            }

        }


        double fps = cv::getTickFrequency() / ((double)cv::getTickCount() - t);  // Calculate Frames per second (FPS)

        // Display FPS 
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "==========================" << std::endl;
        std::cout << std::endl;
        
       std::string frameLabel = "FPS: " + std::to_string(fps);
        // std::string frameLabel = "FrameCount: " + std::to_string(frameCount);
        cv::putText(currentFrame, frameLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);


        cv::imshow("demo", currentFrame);          // Display result.
       outputVideo.write(currentFrame);
        if (cv::waitKey(30) == 'q')         // Exit if 'q' pressed.
        {
            break;
        }


        frameCount++;
    }



    cv::destroyWindow("demo");
    CvCapture.release();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    return 0;
}



// 运动检测线程函数
void motionDetectionThread(MotionDetector& detector) 
{
    while (true) 
    {

    }
}
