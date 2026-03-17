from ultralytics import YOLO

# This is the pretrained nano classification model.
model = YOLO('yolov8n-cls.pt') 

results = model.train(
    data='Sorted Classification Dataset',
    name='CarRimClassifier',
    epochs=25,
    imgsz=224,
    batch=6,

    # These are our data augmentation hyperparameters
    augment=True,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.25,

    # These are the learning rate scheduler hyperparameters
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3.0,
    cos_lr=True
)

model = YOLO('runs/classify/CarRimClassifier/weights/best.pt')
metrics = model.val(split='test') 
print(f"Test Accuracy: {metrics.top1:.2%}")
