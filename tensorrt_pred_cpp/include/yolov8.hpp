//
// Created by ubuntu on 1/20/23.
//
#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
using namespace det;

#define TRT_10 // 如果tensorRT版本低于10则注释。

// 计算IoU
static float calculate_iou(const cv::Rect &rect1, const cv::Rect &rect2) {
    int x1 = (std::max)(rect1.x, rect2.x);
    int y1 = (std::max)(rect1.y, rect2.y);
    int x2 = (std::min)(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = (std::min)(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    float inter_area = (x2 - x1) * (y2 - y1);
    float area1 = rect1.area();
    float area2 = rect2.area();
    float union_area = area1 + area2 - inter_area;

    return inter_area / union_area;
}

// NMS函数
static void nms(std::vector<Object> &objs, float iou_threshold = 0.45f) {
    if (objs.empty()) return;

    // 按置信度降序排序
    std::sort(objs.begin(), objs.end(),
        [](const Object &a, const Object &b) {
            return a.prob > b.prob;
        });

    std::vector<bool> keep(objs.size(), true);
    std::vector<Object> nms_results;

    for (size_t i = 0; i < objs.size(); i++) {
        if (!keep[i]) continue;

        nms_results.push_back(objs[i]);

        for (size_t j = i + 1; j < objs.size(); j++) {
            if (!keep[j]) continue;

            // 只对相同类别的框进行NMS
            // if (objs[i].label != objs[j].label) continue;

            float iou = calculate_iou(objs[i].rect, objs[j].rect);
            if (iou > iou_threshold) {
                keep[j] = false;
            }
        }
    }

    objs = std::move(nms_results);
}

class YOLOv8 {
public:
    explicit YOLOv8(const std::string &engine_file_path);
    ~YOLOv8();

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat &image);
    void                 copy_from_Mat(const cv::Mat &image, cv::Size &size);
    void                 letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);
    void                 infer();
    void                 postprocess(std::vector<Object> &objs);
    static void          draw_objects(const cv::Mat &image,
        cv::Mat &res,
        const std::vector<Object> &objs,
        const std::vector<std::string> &CLASS_NAMES,
        const std::vector<std::vector<unsigned int>> &COLORS);
    int                  num_bindings;
    int                  num_inputs = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *>   host_ptrs;
    std::vector<void *>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t                 stream = nullptr;
    Logger                       gLogger{ nvinfer1::ILogger::Severity::kERROR };
};

YOLOv8::YOLOv8(const std::string &engine_file_path) {
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    // cudaStreamCreate(&this->stream);
    cudaError_t stream_status = cudaStreamCreate(&this->stream);
    assert(stream_status == cudaSuccess);

#ifdef TRT_10
    this->num_bindings = this->engine->getNbIOTensors();
#else
    this->num_bindings = this->engine->getNbBindings();
#endif

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;
#ifdef TRT_10
        std::string        name = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name = this->engine->getBindingName(i);
#endif
        binding.name = name;
        binding.dsize = type_to_size(dtype);
#ifdef TRT_10
        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = engine->bindingIsInput(i);
#endif
        if (IsInput) {
            this->num_inputs += 1;
#ifdef TRT_10
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setInputShape(name.c_str(), dims);
#else
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        }
        else {
#ifdef TRT_10
            dims = this->context->getTensorShape(name.c_str());
#else
            dims = this->context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8() {
#ifdef TRT_10
    delete this->context;
    delete this->engine;
    delete this->runtime;
#else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#endif
    cudaStreamDestroy(this->stream);
    for (auto &ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8::make_pipe(bool warmup) {

    for (auto &bindings : this->input_bindings) {
        void *d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    for (auto &bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;

        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (size_t idx = 0; idx < this->input_bindings.size(); ++idx) {
                auto &bindings = this->input_bindings[idx];
                size_t size = bindings.size * bindings.dsize;
                void *h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[idx], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {
    const float inp_h = size.height;
    const float inp_w = size.width;
    float       height = image.rows;
    float       width = image.cols;

    float r = (std::min)(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh));
    int bottom = int(std::round(dh));
    int left = int(std::round(dw));
    int right = int(std::round(dw));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 });

    // out.create({ 1, 3, (int)inp_h, (int)inp_w }, CV_32F);

    // std::vector<cv::Mat> channels;
    // cv::split(tmp, channels);

    // cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float *)out.data);
    // cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w);
    // cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w * 2);

    // channels[2].convertTo(c0, CV_32F, 1.0f / 255.0f);  // R -> c0
    // channels[1].convertTo(c1, CV_32F, 1.0f / 255.0f);  // G -> c1  
    // channels[0].convertTo(c2, CV_32F, 1.0f / 255.0f);  // B -> c2

    out.create({ 1, 3, size.height, size.width }, CV_32F);

    float *ptr_r = out.ptr<float>(0, 0);  // R通道
    float *ptr_g = out.ptr<float>(0, 1);  // G通道
    float *ptr_b = out.ptr<float>(0, 2);  // B通道

    // 手动复制数据
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            cv::Vec3b pixel = tmp.at<cv::Vec3b>(i, j);
            ptr_r[i * size.width + j] = pixel[2] / 255.0f;  // R
            ptr_g[i * size.width + j] = pixel[1] / 255.0f;  // G
            ptr_b[i * size.width + j] = pixel[0] / 255.0f;  // B
        }
    }

    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;

}

void YOLOv8::copy_from_Mat(const cv::Mat &image) {
    cv::Mat  nchw;
    auto &in_binding = this->input_bindings[0];
    int      width = in_binding.dims.d[3];
    int      height = in_binding.dims.d[2];
    cv::Size size{ width, height };
    this->letterbox(image, nchw, size);


    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{ 4, {1, 3, height, width} });
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{ 4, {1, 3, height, width} });
#endif
}

void YOLOv8::copy_from_Mat(const cv::Mat &image, cv::Size &size) {
    cv::Mat nchw;
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{ 4, {1, 3, size.height, size.width} });
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{ 4, {1, 3, size.height, size.width} });
#endif
}

void YOLOv8::infer() {
#ifdef TRT_10
    this->context->enqueueV3(this->stream);
#else
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

// 后处理
void YOLOv8::postprocess(std::vector<Object> &objs) {
    objs.clear();

    float *output_data = static_cast<float *>(this->host_ptrs[0]);
    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;

    const int NUM_CLASSES = 12;

    auto &output_binding = this->output_bindings[0];

    int batch_size = output_binding.dims.d[0];  // 1
    int box_features = output_binding.dims.d[1];  // 16 (4+12)
    int num_boxes = output_binding.dims.d[2];  // 8400

    float confidence_threshold = 0.25f;

    std::vector<Object> preliminary_objs;

    // 处理每个检测框
    for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
        // 找到最大置信度的类别
        float max_score = 0;
        int class_id = -1;

        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = output_data[(4 + c) * num_boxes + box_idx];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score < confidence_threshold || class_id < 0) continue;

        float cx = output_data[0 * num_boxes + box_idx];
        float cy = output_data[1 * num_boxes + box_idx];
        float w = output_data[2 * num_boxes + box_idx];
        float h = output_data[3 * num_boxes + box_idx];

        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        x1 = (x1 - dw) * ratio;
        y1 = (y1 - dh) * ratio;
        x2 = (x2 - dw) * ratio;
        y2 = (y2 - dh) * ratio;

        x1 = clamp(x1, 0.f, width);
        y1 = clamp(y1, 0.f, height);
        x2 = clamp(x2, 0.f, width);
        y2 = clamp(y2, 0.f, height);

        float box_w = x2 - x1;
        float box_h = y2 - y1;

        if (box_w <= 1.0f || box_h <= 1.0f) continue;

        if (box_w * box_h < 10.0f) continue;

        Object obj;
        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = box_w;
        obj.rect.height = box_h;
        obj.prob = max_score;
        obj.label = class_id;
        preliminary_objs.push_back(obj);
    }

    // 应用NMS
    nms(preliminary_objs, 0.45f);

    // 限制最大检测数量，只允许识别出一个数量。
    const size_t MAX_DETECTIONS = 1;
    objs = preliminary_objs;
    if (objs.size() > MAX_DETECTIONS) {
        objs.resize(MAX_DETECTIONS);
    }
}

// 绘制框
void YOLOv8::draw_objects(const cv::Mat &image,
    cv::Mat &res,
    const std::vector<Object> &objs,
    const std::vector<std::string> &CLASS_NAMES,
    const std::vector<std::vector<unsigned int>> &COLORS) {

    res = image.clone();
    for (auto &obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int      baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), { 0, 0, 255 }, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0, 255, 0 }, 2);
    }
}

#endif  // DETECT_END2END_YOLOV8_HPP