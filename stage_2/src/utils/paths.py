from pathlib import Path

def get_project_root():
    """获取 stage_2 的根目录"""
    # 假设该文件位于 stage_2/src/utils/paths.py
    # parents[0] = utils
    # parents[1] = src
    # parents[2] = stage_2
    return Path(__file__).resolve().parents[2]

def get_stage1_dir(cfg):
    """获取 Stage 1 的数据目录 (输入)"""
    root = get_project_root()
    # 配置文件中可能是相对路径 "../stage_1/data/rendered"
    # 需要将其解析为绝对路径
    path = (root / cfg["dataset"]["input_dir"]).resolve()
    return path

def get_output_dir(cfg):
    """获取 Stage 2 的输出目录"""
    root = get_project_root()
    return root / cfg["dataset"]["output_dir"]
