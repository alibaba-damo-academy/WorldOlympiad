import argparse
import subprocess
from pathlib import Path

from scenecode.code_object.agent_framework.blender.request_lock import (
    acquire_blender_request_lock,
)


def collect_code_py_from_folder(folder_path: Path):
    """收集文件夹下所有 code 目录中的 .py 文件。
    结构: folder_path/<subdir>/code/*.py
    """
    folder_path = Path(folder_path).resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {folder_path}")
    py_files = []
    for subdir in sorted(folder_path.iterdir()):
        if not subdir.is_dir():
            continue
        code_dir = subdir / "code"
        if not code_dir.is_dir():
            continue
        for f in sorted(code_dir.glob("*.py")):
            if f.is_file():
                py_files.append(f)
    return py_files


def run_blender_in_background(code_path: str):
    code_path = Path(code_path).resolve()
    if not code_path.is_file():
        raise FileNotFoundError(f"脚本不存在: {code_path}")

    # 例如: /.../chair_0943_view6/code/chair_0943_view6.py
    # 输出目录: /.../chair_0943_view6/blender_output/
    base_dir = code_path.parent.parent  # .. -> chair_0943_view6
    out_dir = base_dir / "blender_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = code_path.stem  # chair_0943_view6
    blend_path = out_dir / f"{stem}.blend"
    obj_path = out_dir / f"{stem}.obj"

    # pre-run: 运行建模脚本前清空场景
    pre_run_python = "import bpy;bpy.ops.wm.read_factory_settings(use_empty=True)"

    # post-run: 运行建模脚本后保存结果（Blender 3.1+ 使用 wm.obj_export）
    post_run_python = (
        "import bpy;"
        f"bpy.ops.wm.save_as_mainfile(filepath={repr(str(blend_path))});"
        "bpy.ops.wm.obj_export("
        f"filepath={repr(str(obj_path))},"
        "export_selected_objects=False,"
        "export_materials=True,"
        "forward_axis='NEGATIVE_Z',"
        "up_axis='Y'"
        ")"
    )

    # 如果 blender 不在 PATH，请把 `"blender"` 改成绝对路径
    cmd = [
        "blender",
        "-b",  # 后台运行（无界面）
        "--python-expr", pre_run_python,
        "--python", str(code_path),
        "--python-expr", post_run_python,
    ]

    # 后台启动，不阻塞当前终端（你也可以把 stdout/stderr 重定向到日志）
    with acquire_blender_request_lock("code_object.script.run_blender_background"):
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    print(f"已后台启动 Blender：\n  脚本: {code_path}\n  .blend: {blend_path}\n  .obj:   {obj_path}")


def run_batch(folder_path: str):
    """Batch 模式：对文件夹下所有 code 目录中的 .py 文件依次后台运行 Blender。"""
    folder_path = Path(folder_path).resolve()
    py_files = collect_code_py_from_folder(folder_path)
    if not py_files:
        print(f"在 {folder_path} 下未找到任何 <子目录>/code/*.py 文件，退出。")
        return
    print(f"共找到 {len(py_files)} 个脚本，将依次后台启动 Blender：\n")
    for code_path in py_files:
        run_blender_in_background(str(code_path))


def main():
    parser = argparse.ArgumentParser(
        description="后台模式运行 Blender 脚本并保存结果模型"
    )
    parser.add_argument(
        "path",
        help="单文件模式：Blender 脚本路径；batch 模式（--batch）：根目录路径，会运行其下所有 <子目录>/code/*.py",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch 模式：path 为文件夹路径，运行该文件夹下所有 code 目录中的 .py 文件",
    )
    args = parser.parse_args()
    if args.batch:
        run_batch(args.path)
    else:
        run_blender_in_background(args.path)


if __name__ == "__main__":
    main()
