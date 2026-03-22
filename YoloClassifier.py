from ultralytics import YOLO

# This is the pretrained small [!nano] classification model.
model = YOLO('yolov8s-cls.pt') 

results = model.train(
    data='Sorted Classification Dataset',
    name='CarRimClassifier',
    # freeze=9,
    epochs=100,
    imgsz=224,
    batch=6,

    # These are our data augmentation hyperparameters
    augment=True,
    fliplr=0.5,
    flipud=0.5,
    scale=0.4,
    translate=0.15,
    shear=0.15,
    perspective=0.001,
    erasing=0.3,
    mixup=0.3,
    mosaic=0.0,

    # This one is very important to us, as the wheels are not fixed
    # The wheels can rotate 180 degrees since they're roughly symmetrical
    degrees=180,

    # Photometric variation - for lighting changes in the factory (although we min.)
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.3,

    # These are the learning rate scheduler hyperparameters
    lr0=0.001,
    lrf=0.001,
    warmup_epochs=5,
    patience=20,
    cos_lr=True,
)

model = YOLO('runs/classify/CarRimClassifier/weights/best.pt')
metrics = model.val(split='test') 
print(f"Test Accuracy: {metrics.top1:.2%}")
