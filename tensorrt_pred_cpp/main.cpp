#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>
#include <ctime>

namespace fs = ghc::filesystem;
const size_t CAR_MAX_DETECTIONS = 10; // 小车最大检测数量
const float CONF = 0.3f; // 置信度
const size_t video_fps = 60; // 视频帧率

const std::vector<std::string> CLASS_NAMES = {
    "car"
};

const std::vector<std::vector<unsigned int>> COLORS = {
    { 0, 0, 255}
};

// 图片推理
void infer_from_image(std::string car_file_path, std::string image_path);

// 视频推理
void infer_from_video(std::string car_file_path, std::string video_path);

int main(int argc, char **argv) {
    cudaSetDevice(0);

    infer_from_image("car.trt", "1.png");
    // infer_from_video("car.trt", "1.mp4");

    return 0;
}

void infer_from_image(std::string car_file_path, std::string image_path) {
    auto car = new YOLOv8(car_file_path);
    car->make_pipe(true);

    cv::Mat image = cv::imread(image_path);

    std::vector<Object> car_objs;

    // 小车推理
    car->copy_from_Mat(image);
    car->infer();
    car->postprocess(car_objs, CAR_MAX_DETECTIONS, CONF, 1);

    cv::Mat result;
    car->draw_objects(image, result, car_objs, CLASS_NAMES, COLORS);

    cv::Size res_size(1280, 740); // 结果展示size
    cv::resize(result, result, res_size);

    cv::imshow("image", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    delete car;
}

void infer_from_video(std::string car_file_path, std::string video_path) {
    auto car = new YOLOv8(car_file_path);
    car->make_pipe(true);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Can't open the video file!" << std::endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        clock_t start = clock();

        cap >> frame;
        if (frame.empty()) break;

        // 推理
        std::vector<Object> car_objs;

        // 小车
        car->copy_from_Mat(frame);
        car->infer();
        car->postprocess(car_objs, CAR_MAX_DETECTIONS, CONF, 1);

        cv::Mat res;
        car->draw_objects(frame, res, car_objs, CLASS_NAMES, COLORS);

        cv::Size res_size(1280, 740); // 结果展示size
        cv::resize(res, res, res_size);

        clock_t end = clock();
        double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        std::string fps_text = "FPS:" + std::to_string(1.0f / time);
        cv::putText(res, fps_text, cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 1, { 0, 255, 0 }, 2);

        cv::imshow("video", res);
        cv::waitKey(1000 / video_fps); // fps
    }

    cap.release();
    cv::destroyAllWindows();

    delete car;
}
