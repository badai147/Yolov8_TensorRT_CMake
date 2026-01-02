#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
using namespace det;

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

    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat &image);
    void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);
    void infer();

    void postprocess(std::vector<Object> &objs,
        size_t max_detections,
        float confidence_threshold,
        size_t num_classes);

    static void draw_objects(const cv::Mat &image,
        cv::Mat &res,
        const std::vector<Object> &objs,
        const std::vector<std::string> &CLASS_NAMES,
        const std::vector<std::vector<unsigned int>> &COLORS);

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };
};

// 初始化模型
YOLOv8::YOLOv8(const std::string &engine_file_path) {
    initLibNvInferPlugins(&this->gLogger, "");

    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // 创建TensorRT运行时环境。
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 从内存中反序列化CUDA引擎，然后释放存储引擎数据的内存。
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;

    // 创建执行上下文，用于实际执行推理。
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 创建CUDA流，用于异步执行CUDA操作。
    cudaError_t stream_status = cudaStreamCreate(&this->stream);
    assert(stream_status == cudaSuccess);

    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;

        std::string name = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

        // 输入张量
        if (IsInput) {
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            this->context->setInputShape(name.c_str(), dims);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        }
        else {
            dims = this->context->getTensorShape(name.c_str());
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

// 销毁
YOLOv8::~YOLOv8() {
    delete this->context;
    delete this->engine;
    delete this->runtime;
    cudaStreamDestroy(this->stream);

    for (auto &ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

// 创建管道
void YOLOv8::make_pipe(bool warmup) {

    // 输入绑定
    for (auto &bindings : this->input_bindings) {
        void *d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);

        auto name = bindings.name.c_str();
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
    }

    // 输出绑定
    for (auto &bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;

        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
        auto name = bindings.name.c_str();
        this->context->setTensorAddress(name, d_ptr);
    }

    // 预热十次
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
    }

}

// 预处理：调整输入图像格式
void YOLOv8::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {
    const int target_h = size.height;
    const int target_w = size.width;

    const int src_h = image.rows;
    const int src_w = image.cols;

    // std::cout << target_h << " " << target_w << " " << src_h << " " << src_w << std::endl;

    // 选择长边压缩
    float scale = (std::min)((float)target_h / src_h, (float)target_w / src_w);

    int scaled_w = static_cast<int>(src_w * scale);
    int scaled_h = static_cast<int>(src_h * scale);

    // std::cout << scale << " " << scaled_w << " " << scaled_h << std::endl;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(scaled_w, scaled_h));

    int pad_w = target_w - scaled_w;   // 宽度方向总填充
    int pad_h = target_h - scaled_h;   // 高度方向总填充

    int left = pad_w / 2;              // 左边填充
    int right = pad_w - left;          // 右边填充
    int top = pad_h / 2;               // 上边填充  
    int bottom = pad_h - top;          // 下边填充

    // std::cout << left << " " << right << " " << top << " " << bottom << std::endl;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)); // 填充

    out.create({ 1, 3, target_h, target_w }, CV_32F);

    // 通道重排和归一化
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    float *out_data = out.ptr<float>();
    int spatial_size = target_h * target_w;

    for (int i = 0; i < target_h; i++) {
        uchar *row_ptr = rgb.ptr<uchar>(i);
        for (int j = 0; j < target_w; j++) {
            // R通道
            out_data[j + i * target_w] = row_ptr[j * 3 + 2] / 255.0f;
            // G通道  
            out_data[spatial_size + j + i * target_w] = row_ptr[j * 3 + 1] / 255.0f;
            // B通道
            out_data[2 * spatial_size + j + i * target_w] = row_ptr[j * 3 + 0] / 255.0f;
        }
    }

    this->pparam.scale = scale;           // 缩放比例
    this->pparam.pad_left = left;         // 左边填充
    this->pparam.pad_top = top;           // 上边填充
    this->pparam.scaled_width = scaled_w; // 缩放后宽度
    this->pparam.scaled_height = scaled_h;// 缩放后高度
    this->pparam.src_width = src_w;       // 原始宽度
    this->pparam.src_height = src_h;      // 原始高度

}

// 将图像复制到GPU内存
void YOLOv8::copy_from_Mat(const cv::Mat &image) {
    cv::Mat nchw;
    auto &in_binding = this->input_bindings[0];
    int width = in_binding.dims.d[3];
    int height = in_binding.dims.d[2];
    cv::Size size{ width, height };
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{ 4, {1, 3, height, width} });
    this->context->setTensorAddress(name, this->device_ptrs[0]);
}

// 执行推理
void YOLOv8::infer() {
    this->context->enqueueV3(this->stream);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

// 后处理 
void YOLOv8::postprocess(std::vector<Object> &objs,
    size_t max_detections,
    float confidence_threshold,
    size_t num_classes) {

    objs.clear();

    float *output_data = static_cast<float *>(this->host_ptrs[0]);

    auto &scale = this->pparam.scale;           // 缩放比例
    auto &pad_left = this->pparam.pad_left;     // 左边填充
    auto &pad_top = this->pparam.pad_top;       // 上边填充
    auto &src_w = this->pparam.src_width;       // 原始宽度
    auto &src_h = this->pparam.src_height;      // 原始高度

    auto &in_binding = this->input_bindings[0];
    int model_w = in_binding.dims.d[3];
    int model_h = in_binding.dims.d[2];

    auto &output_binding = this->output_bindings[0];

    int batch_size = output_binding.dims.d[0];
    int box_features = output_binding.dims.d[1];
    int num_boxes = output_binding.dims.d[2];

    // std::cout << batch_size << " " << box_features << " " << num_boxes << " " << std::endl;

    std::vector<Object> preliminary_objs;

    // 处理每个检测框
    for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
        // 找到最大置信度的类别
        float max_score = 0;
        int class_id = -1;

        for (int c = 0; c < num_classes; c++) {
            float score = output_data[(4 + c) * num_boxes + box_idx];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score < confidence_threshold || class_id < 0) continue;

        float cx = output_data[0 * num_boxes + box_idx]; // 边界框中心点的x坐标
        float cy = output_data[1 * num_boxes + box_idx];
        float w = output_data[2 * num_boxes + box_idx]; // 边界框的宽度
        float h = output_data[3 * num_boxes + box_idx];

        // 反缩放为原图尺寸
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        x1 = (x1 - pad_left) / scale;
        y1 = (y1 - pad_top) / scale;
        x2 = (x2 - pad_left) / scale;
        y2 = (y2 - pad_top) / scale;

        x1 = clamp(x1, 0.f, src_w);
        y1 = clamp(y1, 0.f, src_h);
        x2 = clamp(x2, 0.f, src_w);
        y2 = clamp(y2, 0.f, src_h);

        float box_w = x2 - x1;
        float box_h = y2 - y1;

        // 过滤小边框
        if (box_w <= 1.0f || box_h <= 1.0f || box_w * box_h < 5.0f) continue;

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

    // 限制最大检测数量。
    objs = preliminary_objs;
    if (objs.size() > max_detections) {
        objs.resize(max_detections);
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

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), color, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0, 255, 0 }, 2);
    }
}

#endif  // DETECT_END2END_YOLOV8_HPP