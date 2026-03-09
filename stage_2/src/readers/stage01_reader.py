from pathlib import Path
from src.utils.io import load_jsonl

class Stage1Reader:
    """
    负责读取 Stage 1 的输出数据。
    它不关心图像内容，只负责把 meta.jsonl 解析出来，
    并把里面的相对路径转换成当前系统可读的绝对路径。
    """
    def __init__(self, stage1_root):
        self.root = Path(stage1_root)
        
    def get_all_branches(self):
        """
        获取所有树枝的 ID (基于文件夹名)
        返回: list of str
        """
        # 遍历根目录下所有子目录，只要包含 meta.jsonl 或 meta.json 就认为是有效的树枝目录
        branches = []
        for p in self.root.iterdir():
            if p.is_dir() and ((p / "meta.jsonl").exists() or (p / "meta.json").exists()):
                branches.append(p.name)
        return sorted(branches)

    def read_branch_meta(self, branch_id):
        """
        读取指定树枝的所有视角元数据
        返回: list of dict
        """
        meta_path_jsonl = self.root / branch_id / "meta.jsonl"
        meta_path_json = self.root / branch_id / "meta.json"
        
        if meta_path_jsonl.exists():
            records = load_jsonl(meta_path_jsonl)
        elif meta_path_json.exists():
            import json
            with open(meta_path_json, 'r', encoding='utf-8') as f:
                records = json.load(f)
        else:
            return []
        
        # 修正路径：将相对路径转为绝对路径
        for r in records:
            # Stage 1 的 image_paths 可能是 {"mask": "...", "normal": "..."}
            # 也可能是旧版的 image_path 字符串
            # 我们统一处理
            
            if "image_paths" in r:
                for k, v in r["image_paths"].items():
                    r["image_paths"][k] = str(self.root / v)
            elif "image_path" in r:
                # 兼容旧版格式
                # 对于 fibonacci_rendered 数据，image_path 可能是 "stage_1/data/..." 这样的长路径
                # 我们取文件名，并假设它在 branch 目录下
                img_name = Path(r["image_path"]).name
                r["image_paths"] = {"mask": str(self.root / branch_id / img_name)}
                
        return records
