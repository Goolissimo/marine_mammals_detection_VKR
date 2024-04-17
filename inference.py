from ultralytics import YOLO

# Загрузка модели
model = YOLO("../best.pt")
#model.to('cuda') #переключение на GPU, если доступно

# Пути к изображениям
img1 = '../dataset/happy_whale_dolphins/test_images/0a0a640daf4634.jpg'
img2 = '../dataset/NDD/BELOW/7.jpg'
img3 = '../dataset/happy_whale_dolphins/test_images/31a4c316bdb3ba.jpg'
img4 = '../dataset/NDD/ABOVE/1787.jpg'
img5 = '../dataset/NDD/ABOVE/1929.jpg'

# Получение предсказаний
results = model([img1, img2, img3, img4, img5])  # Возвращает список с предсказаниями

# Отображение результатов
for idx, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.show()  # display to screen
#    result.save(filename=f'C:/Users/Sava/Pictures/Saved Pictures/result{idx}.jpg')  # Сохранение на диск
