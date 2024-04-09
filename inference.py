from ultralytics import YOLO

# Загружаем нашу модель
model = YOLO("../best.pt")

# Run inference on an image
results = model("../test_image.png", show=False)  # results list
print(results[0].boxes.cls)

# View results
#for r in results:
#    print(r.boxes)  # print the Boxes object containing the detection bounding boxes
#r.show()
