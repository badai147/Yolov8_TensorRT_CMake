import yaml
import tensorrt as trt
from ultralytics import YOLO

def train():
    model = YOLO("./models/12-23-11-54.pt")
    model.train(
        data="armor.yaml", 
        epochs=5,
        batch=8,
        imgsz=640, 
        val=True,
        device='cuda',
        
        hsv_h=0.01,
        hsv_s=0.0,
        hsv_v=0.4,
        degrees=0.5,
        translate=0.05,
        scale=0.0,
        mosaic=0.0,
        fliplr=0.0,
        flipud=0.0
        )

def test():
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    with open('armor.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    if 'test' not in data_config or not data_config['test']:
        print("警告: data.yaml中没有配置测试集路径")
        return
    
    print("正在评估测试集...")
    
    results = model.val(
        data='armor.yaml',
        split='test',  
        batch=8,
        imgsz=640,
        conf=0.001,
        iou=0.65,
        project='runs/val',
        name='test_final',
        exist_ok=True,
        verbose=True,
        device='cuda'
    )
    
    return results

def get_onnx(model):
    "半精度onnx导出"
    model.export(format='onnx', half=True)

def get_tensorRT(onnx_file_path, trt_model_path, 
                 max_workspace_size=1 << 30, fp16_mode=True):
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    input_name = "images"  # TODO
    min_shape = (1, 3, 640, 640)
    opt_shape = (1, 3, 640, 640)
    max_shape = (1, 3, 640, 640)
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build serialized engine.")
        return None

    with open(trt_model_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved as {trt_model_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

if __name__ == "__main__":
    "训练"
    # train()
    
    "测试"
    # test()
    
    "pt转换onnx"
    # model = YOLO('./models/12-23-13-52.pt')
    # get_onnx(model)
    
    "onnx转换tensorRT"
    get_tensorRT('./models/12-23-13-52.onnx', './models/armor.trt')