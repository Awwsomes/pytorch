class Path:
    def __init__(self):
        self.raw_data_root_path = r"D:\A_myData\RC26-Vision\dataset\A_car\raw_data\2026_3_28"
        self.output_root_path = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_28"

class Settings:
    def __init__(self):
        self.start_idx = 0
        self.test_data_start_idx = 1

class ClassConfig:
    def __init__(self):
        self.model_path = r"D:\A_myData\RC26-Vision\Pytorch\yolo11\runs\classify\卷轴分类1_仿真+现实_32类\weights\best.pt"
        self.start_idx = 0
        self.img_root_name = "juanZhou_car"
        self.label_name_list =  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                  "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"]
        self.model_predict_output_dir = r"D:\A_myData\RC26-Vision\Pytorch\yolo11\runs\classify"

class DetectConfig:
    def __init__(self):
        self.model_path = r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\train\角点检测5\weights\best.pt"
        self.data_yaml = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\dataset_yaml\corner10.yaml"
        self.conf_thres = 0.3
        self.iou_thres = 0
        self.max_det = 64

class Config:
    def __init__(self):
        self.path = Path()
        self.settings = Settings()
        self.class_config = ClassConfig()
        self.detect_config = DetectConfig()

config = Config()