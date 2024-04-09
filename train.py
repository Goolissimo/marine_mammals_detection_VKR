from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolov8s.pt")  # load a pretrained model
model.to('cuda')

# Дообучение
model.train(data="../dataset/data.yaml", epochs=100, patience=10, batch=-1, amp=False, plots=False)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
