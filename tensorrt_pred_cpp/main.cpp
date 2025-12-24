#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "B1", "B2", "B3", "B4", "B5", "B7",
    "R1", "R2", "R3", "R4", "R5", "R7"
};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
};

// 推理图片
void infer_from_image(std::string engine_file_path, std::string image_path);

// 推理视频
void infer_from_video(std::string engine_file_path, std::string video_path);

int main(int argc, char **argv) {
    cudaSetDevice(0);

    /*
    objs使用说明：
        std::cout << objs.size() << std::endl;
        std::cout << objs[0].rect.x << std::endl;
    限制：objs只有一个obj：
        obj.rect.x; 检测框的x
        obj.rect.y; 检测框的y
        obj.rect.width; 检测框的宽度
        obj.rect.height; 检测框的高度
        obj.prob; 检测出来的置信度
        obj.label; 检测出来的标签
        CLASS_NAMES[obj.label] 检测出来的类别
    */

    // infer_from_image("12-23-13-52.trt", "000001.jpg");
    // infer_from_video("12-23-13-52.trt", "12-24.mp4");

    return 0;
}

void infer_from_image(std::string engine_file_path, std::string image_path) {
    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    cv::Mat image = cv::imread(image_path);
    cv::Mat image_resized;

    std::vector<Object> objs;
    cv::Size size(640, 640);

    cv::resize(image, image_resized, size);

    yolov8->copy_from_Mat(image_resized, size);
    yolov8->infer();
    yolov8->postprocess(objs);

    cv::Mat res; // 结果矩阵
    yolov8->draw_objects(image_resized, res, objs, CLASS_NAMES, COLORS);

    cv::imshow("result", res);
    cv::waitKey(0);
    cv::destroyAllWindows();

    delete yolov8;
}

void infer_from_video(std::string engine_file_path, std::string video_path) {
    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    cv::VideoCapture cap(video_path);

    cv::Mat frame, frame_resized;
    while (true) {
        cap >> frame;

        if (frame.empty()) break;

        std::vector<Object> objs;
        cv::Size size(640, 640);

        cv::resize(frame, frame_resized, size);

        yolov8->copy_from_Mat(frame_resized, size);
        yolov8->infer();
        yolov8->postprocess(objs);

        cv::Mat res;
        yolov8->draw_objects(frame_resized, res, objs, CLASS_NAMES, COLORS);

        cv::imshow("result", res);

        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

    delete yolov8;
}

