import warnings

warnings.filterwarnings("ignore")
from models import YOLO

if __name__ == "__main__":
    model = YOLO(model=r".\models\cfg\models\11\yolo11.yaml", task="detect")
    model.train(
        data=r".\Grasp-3signs.yaml",
        imgsz=640,
        epochs=300,
        batch=8,
        workers=0,
        device=0,
        optimizer="SGD",
        close_mosaic=10,
        resume=False,
        project="",
        name="",
        single_cls=False,
        cache=False,
        amp=False,
    )
