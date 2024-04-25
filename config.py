#该类用于配置模型
from pathlib import Path
import sys

#获取当前文件的绝对路径
file_path = Path(__file__).resolve()

#获取当前文件的上一级目录的路径
root_path = file_path.parent

#如果当前文件的父目录不在搜索路径中则添加进去
if root_path not in sys.path:
    sys.path.append(str(root_path))

#获取当前项目(工作目录)的相对路径
ROOT = root_path.relative_to(Path.cwd())
#数据源
SOURCES_LIST = ["图像", "摄像头","视频"]#"网络摄像头"

# 模型路径配置
DETECTION_MODEL_DIR = ROOT / 'weights'

#侧边栏模型选择列表
DETECTION_MODEL_LIST = [
    "best.pt"]

categories_map = {
    "Blasst": "稻瘟病",
    "Normal": "健康",
    "Blight": "枯萎病",
    "Brown Spot": "褐斑病",
    "Dead Heart": "枯心病",
    "Downy": "露珠病",
    "False": "假烟病",
    "sheath blight": "鞘病",
    "Streak": "叶纹病",
    "Tungro": "东南亚稻田病或水稻东格鲁病毒病"}
