import json
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


WEB_DIR = Path(__file__).resolve().parent
RECACC_DIR = WEB_DIR.parent
WORKSPACE_ROOT = RECACC_DIR.parent
NOTEBOOK_DIR = RECACC_DIR / "notebooks"
RUN_BASE_DIR = RECACC_DIR / "log" / "notebook_runs" / "web_runs"
MAX_WEB_RUNS = 5


JOB_LOCK = threading.Lock()
CURRENT_PROC = None
CURRENT_EXECUTOR = None
JOB_STATE = {
    "job_id": 0,
    "running": False,
    "done": False,
    "success": False,
    "current_step": "idle",
    "steps": {"s1": "idle", "s2": "idle", "s3": "idle", "s4": "idle"},
    "started_at": None,
    "finished_at": None,
    "logs": [],
    "step_logs": {"s1": [], "s2": [], "s3": [], "s4": []},
    "error": "",
    "run_dir": "",
    "stop_requested": False,
}


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with JOB_LOCK:
        JOB_STATE["logs"].append(line)
        JOB_STATE["logs"] = JOB_STATE["logs"][-2000:]
        step = JOB_STATE.get("current_step", "idle")
        if step in JOB_STATE.get("step_logs", {}):
            JOB_STATE["step_logs"][step].append(line)
            JOB_STATE["step_logs"][step] = JOB_STATE["step_logs"][step][-500:]


def _set_step(step: str, state: str):
    with JOB_LOCK:
        JOB_STATE["current_step"] = step
        JOB_STATE["steps"][step] = state


def _reset_job() -> int:
    with JOB_LOCK:
        next_job_id = int(JOB_STATE.get("job_id", 0)) + 1
        JOB_STATE.update(
            {
                "job_id": next_job_id,
                "running": True,
                "done": False,
                "success": False,
                "current_step": "s1",
                "steps": {"s1": "idle", "s2": "idle", "s3": "idle", "s4": "idle"},
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "logs": [],
                "step_logs": {"s1": [], "s2": [], "s3": [], "s4": []},
                "error": "",
                "run_dir": "",
                "stop_requested": False,
            }
        )
        return next_job_id


def _clear_job_state():
    with JOB_LOCK:
        next_job_id = int(JOB_STATE.get("job_id", 0)) + 1
        JOB_STATE.update(
            {
                "job_id": next_job_id,
                "running": False,
                "done": False,
                "success": False,
                "current_step": "idle",
                "steps": {"s1": "idle", "s2": "idle", "s3": "idle", "s4": "idle"},
                "started_at": None,
                "finished_at": None,
                "logs": [],
                "step_logs": {"s1": [], "s2": [], "s3": [], "s4": []},
                "error": "",
                "run_dir": "",
                "stop_requested": False,
            }
        )


def _mark_job_stopped(reason: str = "任务已被用户终止"):
    with JOB_LOCK:
        next_job_id = int(JOB_STATE.get("job_id", 0)) + 1
        JOB_STATE["job_id"] = next_job_id
        JOB_STATE["running"] = False
        JOB_STATE["done"] = True
        JOB_STATE["success"] = False
        JOB_STATE["finished_at"] = datetime.now().isoformat()
        JOB_STATE["error"] = reason
        cur = JOB_STATE.get("current_step", "s1")
        if cur in JOB_STATE.get("steps", {}):
            JOB_STATE["steps"][cur] = "error"
        JOB_STATE["stop_requested"] = True


def _is_stop_requested() -> bool:
    with JOB_LOCK:
        return bool(JOB_STATE.get("stop_requested", False))


def _request_stop():
    global CURRENT_PROC, CURRENT_EXECUTOR
    with JOB_LOCK:
        JOB_STATE["stop_requested"] = True

    ep = CURRENT_EXECUTOR
    if ep is not None:
        try:
            if getattr(ep, "kc", None) is not None:
                ep.kc.interrupt_kernel()
        except Exception:
            pass
        try:
            if getattr(ep, "km", None) is not None:
                ep.km.shutdown_kernel(now=True)
        except Exception:
            pass

    proc = CURRENT_PROC
    if proc is not None and proc.poll() is None:
        _log("收到终止请求，正在停止当前Notebook执行进程")
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _kill_all_notebook_kernels():
    """Best-effort kill for all local notebook kernels (ipykernel) on Windows."""
    try:
        probe = subprocess.run(
            [
                "wmic",
                "process",
                "where",
                "CommandLine like '%ipykernel_launcher%'",
                "get",
                "ProcessId,CommandLine",
                "/FORMAT:CSV",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        )
    except Exception as exc:
        _log(f"扫描kernel进程失败: {exc}")
        return

    pids = []
    for raw in (probe.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if not parts:
            continue
        pid = parts[-1]
        if pid.isdigit():
            pids.append(int(pid))

    if not pids:
        _log("未发现可清理的notebook kernel进程")
        return

    current_pid = os.getpid()
    killed = 0
    for pid in sorted(set(pids)):
        if pid == current_pid:
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
            )
            killed += 1
        except Exception:
            pass

    _log(f"已清理notebook kernel进程: {killed} 个")


def _iter_cell_output_lines(cell: dict):
    outputs = cell.get("outputs", []) or []
    for out in outputs:
        otype = out.get("output_type")
        if otype == "stream":
            text = out.get("text", "")
            for line in str(text).splitlines():
                if line.strip():
                    yield line
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {}) or {}
            text = data.get("text/plain") or data.get("text/markdown") or ""
            if isinstance(text, list):
                text = "".join(text)
            for line in str(text).splitlines():
                if line.strip():
                    yield line
        elif otype == "error":
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            yield f"{ename}: {evalue}"
            for line in out.get("traceback", []) or []:
                if line.strip():
                    yield line


class _LoggingExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, *args, logger=None, stop_checker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logger or (lambda _msg: None)
        self._stop_checker = stop_checker or (lambda: False)

    def preprocess_cell(self, cell, resources, cell_index):
        if self._stop_checker():
            raise RuntimeError("任务已被用户终止")

        ctype = cell.get("cell_type", "")
        if ctype == "code":
            self._logger(f"[Cell {cell_index + 1}] 开始执行")

        out_cell, resources = super().preprocess_cell(cell, resources, cell_index)

        if ctype == "code":
            has_output = False
            for line in _iter_cell_output_lines(out_cell):
                has_output = True
                self._logger(f"[Cell {cell_index + 1}] {line}")
            if not has_output:
                self._logger(f"[Cell {cell_index + 1}] (无输出)")

        if self._stop_checker():
            raise RuntimeError("任务已被用户终止")
        return out_cell, resources


def _finish_job(success: bool, error: str = "", job_id: int | None = None):
    with JOB_LOCK:
        if job_id is not None and int(JOB_STATE.get("job_id", -1)) != int(job_id):
            return
        JOB_STATE["running"] = False
        JOB_STATE["done"] = True
        JOB_STATE["success"] = success
        JOB_STATE["finished_at"] = datetime.now().isoformat()
        JOB_STATE["error"] = error


def _replace_line(lines, prefix: str, new_line: str):
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = new_line
            return True
    return False


def _prune_old_web_runs(base_dir: Path, keep: int = MAX_WEB_RUNS):
    if keep <= 0:
        return

    base_dir.mkdir(parents=True, exist_ok=True)
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if len(dirs) <= keep:
        return

    # Run目录名使用时间戳，按目录名降序可稳定保留最新结果。
    dirs_sorted = sorted(dirs, key=lambda p: p.name, reverse=True)
    stale_dirs = dirs_sorted[keep:]
    for d in stale_dirs:
        shutil.rmtree(d, ignore_errors=True)
        _log(f"已清理旧运行目录: {d}")


def _patch_training_notebook(path: Path, config: dict):
    nb = nbformat.read(path, as_version=4)
    train_epochs = int(config.get("train_epochs", 3))
    train_lr = float(config.get("train_lr", 1e-3))

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = list(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "").splitlines(True)
        hit = _replace_line(src, "EPOCHS =", f"EPOCHS = {train_epochs}\n")
        hit = _replace_line(src, "LR =", f"LR = {train_lr}\n") or hit
        if hit:
            cell["source"] = src
            break

    nbformat.write(nb, path)


def _patch_evaluate_notebook(path: Path, config: dict):
    nb = nbformat.read(path, as_version=4)

    mode = config.get("mode", "small")
    models = config.get("models", ["MLP", "VAE", "NCF"])
    n_train = int(config.get("n_train", 12))
    n_val = int(config.get("n_val", 64))
    utility_epochs = int(config.get("utility_epochs", 1))
    mc_iters = int(config.get("mc_iters", 4))
    lr = float(config.get("lr", 1e-3))
    seeds = config.get("seeds", [7, 42, 99])
    if not isinstance(seeds, list):
        seeds = [42]

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        if "EXPERIMENT_MODE" not in text:
            continue

        src = list(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "").splitlines(True)
        _replace_line(src, "EXPERIMENT_MODE =", f'EXPERIMENT_MODE = "{mode}"\n')
        _replace_line(src, "BENCH_MODELS =", f"BENCH_MODELS = {repr(models)}\n")

        _replace_line(src, "SMALL_SEED =", f"SMALL_SEED = {int(seeds[0]) if seeds else 42}\n")
        _replace_line(src, "SMALL_N_TRAIN =", f"SMALL_N_TRAIN = {n_train}\n")
        _replace_line(src, "SMALL_N_VAL =", f"SMALL_N_VAL = {n_val}\n")
        _replace_line(src, "SMALL_EPOCHS_IN_UTILITY =", f"SMALL_EPOCHS_IN_UTILITY = {utility_epochs}\n")
        _replace_line(src, "SMALL_MC_ITERS_PER_SAMPLE =", f"SMALL_MC_ITERS_PER_SAMPLE = {mc_iters}\n")
        _replace_line(src, "SMALL_LR_BENCH =", f"SMALL_LR_BENCH = {lr}\n")

        _replace_line(src, "MS_SEEDS =", f"MS_SEEDS = {repr([int(x) for x in seeds])}\n")
        _replace_line(src, "MS_N_TRAIN =", f"MS_N_TRAIN = {n_train}\n")
        _replace_line(src, "MS_N_VAL =", f"MS_N_VAL = {n_val}\n")
        _replace_line(src, "MS_EPOCHS_IN_UTILITY =", f"MS_EPOCHS_IN_UTILITY = {utility_epochs}\n")
        _replace_line(src, "MS_MC_ITERS_PER_SAMPLE =", f"MS_MC_ITERS_PER_SAMPLE = {mc_iters}\n")
        _replace_line(src, "MS_LR_BENCH =", f"MS_LR_BENCH = {lr}\n")

        cell["source"] = src
        break

    nbformat.write(nb, path)


def _copy_all_notebooks(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in src_dir.glob("*.ipynb"):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
            copied += 1
    _log(f"已复制 {copied} 个notebook到运行目录")


def _run_notebook(nb_path: Path):
    global CURRENT_PROC, CURRENT_EXECUTOR
    if _is_stop_requested():
        raise RuntimeError("任务已被用户终止")

    if importlib.util.find_spec("nbconvert") is None or importlib.util.find_spec("nbclient") is None:
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "nbconvert",
            "nbclient",
            "ipykernel",
        ]
        _log("检测到 nbconvert 未安装，正在安装执行依赖")
        _log("执行命令: " + " ".join(install_cmd))
        proc_install = subprocess.run(
            install_cmd,
            cwd=str(NOTEBOOK_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc_install.stdout:
            for line in proc_install.stdout.splitlines():
                if line.strip():
                    _log(line.strip())
        if proc_install.returncode != 0:
            raise RuntimeError(f"安装 nbconvert 失败, exit={proc_install.returncode}")

    _log(f"执行Notebook: {nb_path.name}")
    nb = nbformat.read(nb_path, as_version=4)
    ep = _LoggingExecutePreprocessor(
        timeout=-1,
        kernel_name="python3",
        logger=_log,
        stop_checker=_is_stop_requested,
    )
    CURRENT_EXECUTOR = ep
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        nbformat.write(nb, nb_path)
    finally:
        CURRENT_EXECUTOR = None
        CURRENT_PROC = None

    if _is_stop_requested():
        raise RuntimeError("任务已被用户终止")


def _pipeline(config: dict, job_id: int):
    try:
        _prune_old_web_runs(RUN_BASE_DIR, keep=MAX_WEB_RUNS - 1)

        if _is_stop_requested():
            raise RuntimeError("任务已被用户终止")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUN_BASE_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        with JOB_LOCK:
            JOB_STATE["run_dir"] = str(run_dir)

        _set_step("s1", "running")
        _log("Step1: 初始化运行目录与参数")

        _copy_all_notebooks(NOTEBOOK_DIR, run_dir)

        train_nb_src = run_dir / "rec_training.ipynb"
        eval_nb_src = run_dir / "evaluate.ipynb"
        train_nb_tmp = run_dir / "rec_training_run.ipynb"
        eval_nb_tmp = run_dir / "evaluate_run.ipynb"

        if not train_nb_src.exists() or not eval_nb_src.exists():
            raise RuntimeError("运行目录中缺少 rec_training.ipynb 或 evaluate.ipynb")

        shutil.copy2(train_nb_src, train_nb_tmp)
        shutil.copy2(eval_nb_src, eval_nb_tmp)

        _patch_training_notebook(train_nb_tmp, config)
        _patch_evaluate_notebook(eval_nb_tmp, config)

        _set_step("s1", "done")
        _log("Step1 完成")

        if _is_stop_requested():
            raise RuntimeError("任务已被用户终止")

        _set_step("s2", "running")
        _log("Step2: 执行训练笔记本")
        _run_notebook(train_nb_tmp)
        _set_step("s2", "done")
        _log("Step2 完成")

        if _is_stop_requested():
            raise RuntimeError("任务已被用户终止")

        _set_step("s3", "running")
        _log("Step3/4: 执行评估笔记本")
        _run_notebook(eval_nb_tmp)
        _set_step("s3", "done")

        if _is_stop_requested():
            raise RuntimeError("任务已被用户终止")

        _set_step("s4", "running")
        _log("Step4: 汇总结果")
        time.sleep(0.2)
        _set_step("s4", "done")
        _log("Step4 完成，流程结束")

        _finish_job(True, job_id=job_id)
    except Exception as exc:
        _log(f"ERROR: {exc}")
        with JOB_LOCK:
            if int(JOB_STATE.get("job_id", -1)) == int(job_id):
                cur = JOB_STATE.get("current_step", "s1")
                if cur in JOB_STATE["steps"]:
                    JOB_STATE["steps"][cur] = "error"
        _finish_job(False, str(exc), job_id=job_id)


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/" or self.path == "/train_wizard.html":
            return self._serve_file(WEB_DIR / "train_wizard.html", "text/html; charset=utf-8")
        if self.path == "/api/health":
            return self._json(HTTPStatus.OK, {"ok": True})
        if self.path == "/api/status":
            with JOB_LOCK:
                payload = dict(JOB_STATE)
                payload["logs"] = "\n".join(JOB_STATE["logs"])
            return self._json(HTTPStatus.OK, payload)
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        if self.path == "/api/reset":
            _request_stop()
            _kill_all_notebook_kernels()
            _clear_job_state()
            _log("状态已重置")
            return self._json(HTTPStatus.OK, {"ok": True, "message": "reset done"})

        if self.path == "/api/stop":
            with JOB_LOCK:
                if not JOB_STATE["running"]:
                    return self._json(HTTPStatus.CONFLICT, {"ok": False, "error": "no running job"})
            _request_stop()
            _kill_all_notebook_kernels()
            _mark_job_stopped("任务已被用户终止")
            return self._json(HTTPStatus.OK, {"ok": True, "message": "stop requested"})

        if self.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})

        with JOB_LOCK:
            if JOB_STATE["running"]:
                return self._json(HTTPStatus.CONFLICT, {"ok": False, "error": "job already running"})

        job_id = _reset_job()
        _log("收到运行请求")

        t = threading.Thread(target=_pipeline, args=(payload, job_id), daemon=True)
        t.start()

        return self._json(HTTPStatus.OK, {"ok": True, "message": "job started"})


def main():
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
