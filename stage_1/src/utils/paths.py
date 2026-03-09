from pathlib import Path

def project_root() -> Path:
    # 返回 stage_1 的根目录
    # file: stage_1/src/utils/paths.py
    # parents[0]: utils
    # parents[1]: src
    # parents[2]: stage_1
    return Path(__file__).resolve().parents[2]

def mesh_dir(cfg):
    return project_root() / cfg["paths"]["mesh_dir"]

def output_dir(cfg):
    return project_root() / cfg["paths"]["output_dir"]