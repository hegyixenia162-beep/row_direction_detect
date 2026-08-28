"""棉花种植行有限线段人工标注平台的本地 HTTP 服务。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


WORKSPACE = Path(__file__).resolve().parent
WEB_DIR = WORKSPACE / "annotation_tool"


def parse_args() -> argparse.Namespace:
    """解析启动参数；允许更换影像、算法结果和人工标注目录。"""

    parser = argparse.ArgumentParser(description="棉花种植行人工线段标注平台")
    parser.add_argument("--images", default="test_photo", help="待标注图像目录")
    parser.add_argument("--annotations", default="annotations/manual", help="人工标注保存目录")
    parser.add_argument("--predictions", default="outputs", help="算法结果 JSON 目录")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；默认仅本机可访问")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    return parser.parse_args()


class AnnotationHandler(SimpleHTTPRequestHandler):
    """提供静态页面、原始图像、算法结果以及人工标注读写接口。"""

    image_dir: Path
    annotation_dir: Path
    prediction_dir: Path

    def log_message(self, format_string: str, *args: object) -> None:
        """保留简洁访问日志，便于发现保存失败或非法请求。"""

        print(f"[标注服务] {self.address_string()} - {format_string % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        """统一发送 UTF-8 JSON，确保中文文件名和错误信息不会被转义破坏。"""

        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def safe_image_path(self, filename: str) -> Path | None:
        """解析图像文件名并阻止路径穿越，只允许访问配置目录内的直接子文件。"""

        decoded = unquote(filename)
        # 文件名中存在目录分隔符时拒绝请求，避免通过 URL 读取工作区的任意文件。
        if Path(decoded).name != decoded or posixpath.basename(decoded) != decoded:
            return None
        candidate = (self.image_dir / decoded).resolve()
        # 解析后的父目录必须等于影像目录，且文件必须真实存在。
        if candidate.parent != self.image_dir.resolve() or not candidate.is_file():
            return None
        return candidate

    def serve_file(self, path: Path) -> None:
        """流式返回大图，避免一次性把 20–25 MP 文件全部读入服务端内存。"""

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as source:
            # 分块复制文件，为什么使用固定块：兼顾内存占用与大图传输效率。
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def list_images(self) -> list[dict]:
        """返回可标注图像及完成状态，前端据此显示队列和进度。"""

        suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        records: list[dict] = []
        # 按文件名排序可确保上一张/下一张顺序在不同运行间保持稳定。
        for path in sorted(self.image_dir.iterdir()):
            # 仅把支持的普通图像文件加入标注队列。
            if path.is_file() and path.suffix.lower() in suffixes:
                annotation_path = self.annotation_dir / f"{path.stem}_manual.json"
                line_count = 0
                # 已有标注时读取行数；损坏 JSON 不阻断列表接口，而交给打开图像时报告。
                if annotation_path.exists():
                    try:
                        line_count = len(json.loads(annotation_path.read_text(encoding="utf-8")).get("rows", []))
                    except (OSError, json.JSONDecodeError):
                        line_count = 0
                records.append({"name": path.name, "annotated": annotation_path.exists(), "row_count": line_count})
        return records

    def annotation_path(self, image_name: str) -> Path | None:
        """由合法图像名推导人工标注路径，禁止客户端任意指定保存位置。"""

        image_path = self.safe_image_path(image_name)
        # 只有真实存在于图像目录中的文件才能拥有标注文件。
        if image_path is None:
            return None
        return self.annotation_dir / f"{image_path.stem}_manual.json"

    def do_GET(self) -> None:
        """分发页面、图像、人工标注和算法参考结果读取请求。"""

        parsed = urlparse(self.path)
        route = parsed.path
        # 图像列表接口是前端初始化的入口。
        if route == "/api/images":
            self.send_json({"images": self.list_images()})
            return
        # 原图接口按文件名读取，并通过安全路径检查阻止目录穿越。
        if route.startswith("/api/image/"):
            image_path = self.safe_image_path(route.removeprefix("/api/image/"))
            if image_path is None:
                # 使用 UTF-8 JSON 返回中文错误，避免标准 send_error 的 Latin-1 页面编码导致连接中断。
                self.send_json({"error": "图像不存在"}, HTTPStatus.NOT_FOUND)
            else:
                self.serve_file(image_path)
            return
        # 人工标注不存在时返回空结构，前端无需把 404 当作异常处理。
        if route.startswith("/api/annotation/"):
            image_name = unquote(route.removeprefix("/api/annotation/"))
            path = self.annotation_path(image_name)
            if path is None:
                self.send_json({"error": "图像不存在"}, HTTPStatus.NOT_FOUND)
            elif path.exists():
                try:
                    self.send_json(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError) as exc:
                    self.send_json({"error": f"标注文件损坏：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_json({"source": image_name, "rows": []})
            return
        # 算法参考层只读；没有对应结果时返回空行，保持人工标注流程可用。
        if route.startswith("/api/prediction/"):
            image_name = unquote(route.removeprefix("/api/prediction/"))
            image_path = self.safe_image_path(image_name)
            if image_path is None:
                self.send_json({"error": "图像不存在"}, HTTPStatus.NOT_FOUND)
            else:
                prediction_path = self.prediction_dir / f"{image_path.stem}_rows.json"
                if prediction_path.exists():
                    try:
                        self.send_json(json.loads(prediction_path.read_text(encoding="utf-8")))
                    except (OSError, json.JSONDecodeError) as exc:
                        self.send_json({"error": f"算法结果损坏：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                else:
                    self.send_json({"source": image_name, "rows": []})
            return

        # 其余请求映射到前端静态目录，根路径固定返回标注页面。
        relative = "index.html" if route == "/" else route.lstrip("/")
        static_path = (WEB_DIR / relative).resolve()
        # 静态路径必须位于工具目录内，且必须是普通文件。
        if WEB_DIR.resolve() not in static_path.parents or not static_path.is_file():
            # 静态资源错误也统一使用 UTF-8 JSON，保证中文提示能稳定传输。
            self.send_json({"error": "页面不存在"}, HTTPStatus.NOT_FOUND)
        else:
            self.serve_file(static_path)

    def do_POST(self) -> None:
        """校验并保存单张图的人工线段标注。"""

        route = urlparse(self.path).path
        # 当前唯一写接口是人工标注保存，其他 POST 请求全部拒绝。
        if not route.startswith("/api/annotation/"):
            self.send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            return
        image_name = unquote(route.removeprefix("/api/annotation/"))
        target = self.annotation_path(image_name)
        # 不允许为不存在的图像创建孤立标注文件。
        if target is None:
            self.send_json({"error": "图像不存在"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            # 限制请求体大小，为什么：线段 JSON 应很小，可防止意外提交大文件耗尽内存。
            if length <= 0 or length > 5 * 1024 * 1024:
                raise ValueError("请求体大小不合法")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            rows = payload.get("rows")
            # rows 必须是数组，防止前端状态错误覆盖掉已有标注。
            if not isinstance(rows, list):
                raise ValueError("rows 必须是数组")
            # 逐条验证有限端点结构；坐标范围由前端按原图裁剪，服务端保证数值类型和长度。
            for row in rows:
                if not isinstance(row, dict) or not all(
                    isinstance(row.get(key), list) and len(row[key]) == 2 for key in ("start", "end")
                ):
                    raise ValueError("每条线必须包含二维 start 和 end")
                if not all(isinstance(value, (int, float)) for key in ("start", "end") for value in row[key]):
                    raise ValueError("端点坐标必须是数值")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
            self.send_json({"ok": True, "path": str(target), "row_count": len(rows)})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    """配置目录并启动仅绑定本机的多线程标注服务。"""

    args = parse_args()
    image_dir = (WORKSPACE / args.images).resolve()
    annotation_dir = (WORKSPACE / args.annotations).resolve()
    prediction_dir = (WORKSPACE / args.predictions).resolve()
    # 影像目录是平台运行的必要输入，不存在时立即停止并给出明确路径。
    if not image_dir.is_dir():
        raise SystemExit(f"图像目录不存在：{image_dir}")
    annotation_dir.mkdir(parents=True, exist_ok=True)
    AnnotationHandler.image_dir = image_dir
    AnnotationHandler.annotation_dir = annotation_dir
    AnnotationHandler.prediction_dir = prediction_dir
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"标注平台已启动：http://{args.host}:{args.port}")
    print(f"图像目录：{image_dir}")
    print(f"标注目录：{annotation_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止标注平台……")
    finally:
        server.server_close()


# 直接运行脚本时启动服务；被测试代码导入时不占用端口。
if __name__ == "__main__":
    main()
