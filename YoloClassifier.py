from ultralytics import YOLO

# This is the pretrained small [!nano] classification model.
model = YOLO('yolov8s-cls.pt') 

results = model.train(
    data='Sorted Classification Dataset',
    name='CarRimClassifier',
    epochs=25,
    imgsz=224,
    batch=6,

    # These are our data augmentation hyperparameters
    augment=True,
    fliplr=0.5,
    shear=0.25,
    translate=0.1,
    mixup=0.25,

    # This one is very important to us, as the 3-jaw chuck doesn't fix rotation
    degrees=30,

    # These are the learning rate scheduler hyperparameters
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3.0,
    cos_lr=True
)

model = YOLO('runs/classify/CarRimClassifier/weights/best.pt')
metrics = model.val(split='test') 
print(f"Test Accuracy: {metrics.top1:.2%}")
