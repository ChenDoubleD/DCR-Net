import warnings

warnings.filterwarnings('ignore')
from models import YOLO
from models import RTDETR

if __name__ == '__main__':
    model = YOLO(r'.\runs\detect/runs/train\Docking20251222-3signs\rtdetrl-tietu+guang\weights\last.pt')
    model.val(data=r'./data_process/Docking20251222-3signs.yaml',
              imgsz=640,
              batch=16,
              split='test',  # 测试集
              project='runs/val/Docking20251222-3signs',
              name='rtdetrl-tietu+guang-last',
              workers=0,
              device=0,
              )
