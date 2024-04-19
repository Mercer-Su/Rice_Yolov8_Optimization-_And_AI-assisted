from ultralytics import YOLO

if __name__ == '__main__':
    # 直接使用预训练权重或yaml文件创建模型
    # model = YOLO(r'yolov8n.pt')
    # model.train(cfg = 'ultralytics/cfg/default.yam', data=r"D:\Dataset\AutoMine\Middle4-2127\Dataset_NoPedestrain\data_NoPedestrian.yaml", name='train-200epoch-v5InV8-ExistAnchor')

    #使用预训练权重+配置文件创建模型
    model = YOLO(r".\YOLO-v8\ultralytics\cfg\models\v8\yolov8.yaml")
    model.load('yolov8n.pt')
    model.train(cfg='ultralytics/cfg/default.yaml',
                data=r"E:\WorkSpace\Event\Rice\Dataset\data.yaml",
                optimizer='SGD',
                epochs=200,
                project=r".\runs\detect",
                name='train-Rice.yaml',
                )

    # #恢复中断的训练
    # Load a model
    # model = YOLO(r".\YOLO-v8\ultralytics-main\runs\detect\VOC\ttrain-Rice.yaml\weights\last.pt")  # load a partially trained model
    # # Resume training
    # result = model.train(resume=True)

    # # 模型验证
    # model = YOLO(r"E:\WorkSpace\Event\Rice\YOLO-v8\weights\best.pt")
    # model.val( data= r"E:\WorkSpace\Event\Rice\YOLO-v8\dataset\data.yaml",
    #            cfg = 'ultralytics/cfg/default.yaml',
    #            split = 'test',
    #            project=r'E:\WorkSpace\Event\Rice\YOLO-v8\runs\detect',
    #            name='val-500epoch-yolov8n.yaml')

    # 模型推理
    # model = YOLO(r".\weights\best.pt")
    # model.predict(source=r".\Dataset\test\images",
    #               save=True,
    #               project=r'.\runs\detect',
    #               name='predict-Rice'
    #               )