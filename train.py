from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolov8s.pt")  # load a pretrained model
#Переключение на GPU
model.to('cuda')

# Обучение модели
model.train(data='../data.yaml', 
            epochs=100, 
            patience=10, 
            optimizer='auto',
            imgsz=640,
            batch=-1,
            amp=False, 
            plots=False)

#Оценка качества на валидации
metrics = model.val()
