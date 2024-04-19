from ultralytics import YOLO

if __name__ == '__main__':

    # 模型验证
    # model = YOLO(r".\weights\best.pt")
    # model.val( data= r"Dataset\data.yaml",
    #            cfg = 'ultralytics/cfg/default.yaml',
    #            split = 'test',
    #            project=r'.\runs\detect',
    #            name='val-500epoch-yolov8n.yaml')

    # # 模型推理
    model = YOLO(r".\weights\best.pt")
    model.predict(source=r".\Dataset\test\images",
                save=True,
                project=r'.\runs\detect',
                name='predict-Rice'
                 )