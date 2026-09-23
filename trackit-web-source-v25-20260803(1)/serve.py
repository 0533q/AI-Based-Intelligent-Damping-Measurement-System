#!/usr/bin/env python3
"""
振动分析系统开发服务器
用法: python serve.py [端口]   (默认 8000)

与 python -m http.server 的区别:
1. 发送 COOP/COEP 响应头 -> 页面获得 crossOriginIsolated，
   解锁 ONNX Runtime 多线程推理 (SharedArrayBuffer)，1280 推理可提速 2-4 倍
2. 发送 Cache-Control: no-cache -> 避免浏览器缓存旧 JS
3. 正确 MIME 类型
"""
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

COOP_COEP = True  # 置 False 可关闭多线程头


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'application/javascript',
        '.onnx': 'application/octet-stream',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.mp4': 'video/mp4',
    }

    def end_headers(self):
        if COOP_COEP:
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # 静默日志


with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
    print(f"振动分析系统服务已启动: http://localhost:{PORT}  (多线程推理: {'开' if COOP_COEP else '关'})")
    httpd.serve_forever()
