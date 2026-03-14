from ultralytics import YOLO

# This is the pretrained nano classification model.
model = YOLO('yolov8n-cls.pt') 

results = model.train(
    data='Yolo Classification Dataset',
    name='CarRimClassifier'
    epochs=50,
    imgsz=224,
    batch=6,

    # Data augmentation hyperparameters
    augment=True,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,

    # Learning rate scheduler hyperparameters
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3.0,
    cos_lr=True
)

model = YOLO('runs/classify/CarRimClassifier/weights/best.pt')
metrics = model.val(split='test') 
print(f"Test Accuracy: {metrics.top1:.2%}")
