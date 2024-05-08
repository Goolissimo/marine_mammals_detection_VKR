from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics import YOLO


def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A         

            # Insert required transformation here
            T = [
                A.Blur(p=0.1),
                A.MedianBlur(p=0.1),
                A.ToGray(p=0.1),
                A.CLAHE(p=0.1),
                A.RandomBrightnessContrast(p=0.1)
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

Albumentations.__init__ = __init__

model = YOLO('yolov8s.pt')

model.train(data='/home/dev2/dolphin_detection/datav3/data.yaml', epochs=200, patience=10, batch=20, amp=False, plots=True, device=0)
metrics = model.val()
