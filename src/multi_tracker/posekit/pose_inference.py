#!/usr/bin/env python3
"""
Pose inference utilities (subprocess + caching).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
import shutil
import threading
import time
import urllib.request
import urllib.error
import socket
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PoseInferenceService:
    """PoseInferenceService API surface documentation."""
    CACHE_VERSION = 3

    def __init__(
        self,
        out_root: Path,
        keypoint_names: List[str],
        skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    ):
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])
        kpt_sig_src = "|".join(self.keypoint_names) + f":{len(self.keypoint_names)}"
        self.kpt_sig = hashlib.sha1(kpt_sig_src.encode("utf-8")).hexdigest()[:12]
        self._cache_mem: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}

    def _weights_sig(self, weights_path: Path) -> Optional[str]:
        if not weights_path.exists() or not weights_path.is_file():
            return None
        stat = weights_path.stat()
        token = f"{weights_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _sleap_model_sig(self, model_dir: Path) -> Optional[str]:
        if not model_dir.exists() or not model_dir.is_dir():
            return None
        parts = []
        for name in ("best.ckpt", "training_config.yaml"):
            p = model_dir / name
            if p.exists() and p.is_file():
                stat = p.stat()
                parts.append(f"{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}")
        if parts:
            token = "|".join(parts)
            return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        try:
            stat = model_dir.stat()
            token = f"{model_dir.resolve()}|{stat.st_mtime_ns}"
        except Exception:
            token = str(model_dir.resolve())
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _model_sig(self, model_path: Path, backend: str) -> Optional[str]:
        backend = (backend or "yolo").lower()
        if backend == "sleap":
            return self._sleap_model_sig(model_path)
        return self._weights_sig(model_path)

    def _cache_key(self, model_path: Path, backend: str) -> str:
        m = self._model_sig(model_path, backend) or "missing"
        token = (
            f"v{self.CACHE_VERSION}|k{self.kpt_sig}|n{len(self.keypoint_names)}|"
            f"b{backend}|m{m}"
        )
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _cache_dir(self) -> Path:
        return self.out_root / "posekit" / "predictions"

    def _cache_path(self, model_path: Path, backend: str) -> Path:
        return self._cache_dir() / f"{self._cache_key(model_path, backend)}.json"

    def _read_cache_file(
        self, path: Path, backend: str
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            if int(meta.get("version", -1)) != self.CACHE_VERSION:
                return None
            if meta.get("backend") and str(meta.get("backend")) != backend:
                return None
            if "backend" not in meta and backend != "yolo":
                return None
            if meta.get("kpt_sig") != self.kpt_sig:
                return None
            if int(meta.get("num_kpts", -1)) != len(self.keypoint_names):
                return None
            preds = {}
            for k, v in (data.get("preds") or {}).items():
                preds[k] = [(float(x), float(y), float(c)) for x, y, c in v]
            return preds
        except Exception:
            return None

    def load_cache(
        self, model_path: Path, backend: str = "yolo"
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """load_cache method documentation."""
        key = self._cache_key(model_path, backend)
        if key in self._cache_mem:
            return self._cache_mem[key]
        preds = self._read_cache_file(self._cache_path(model_path, backend), backend) or {}
        self._cache_mem[key] = preds
        return preds

    def _write_cache(
        self,
        model_path: Path,
        preds: Dict[str, List[Tuple[float, float, float]]],
        backend: str = "yolo",
    ):
        self._cache_dir().mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self.CACHE_VERSION,
            "kpt_sig": self.kpt_sig,
            "num_kpts": len(self.keypoint_names),
            "backend": backend,
            "model_path": str(model_path),
            "model_sig": self._model_sig(model_path, backend),
        }
        payload = {"meta": meta, "preds": preds}
        path = self._cache_path(model_path, backend)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._cache_mem[self._cache_key(model_path, backend)] = preds

    def merge_cache(self: object, model_path: Path, new_preds: Dict[str, List[Tuple[float, float, float]]], backend: str = 'yolo') -> object:
        """merge_cache method documentation."""
        cache = self.load_cache(model_path, backend)
        cache.update(new_preds)
        self._write_cache(model_path, cache, backend)

    def clear_cache(self, model_path: Optional[Path] = None, backend: str = "yolo") -> int:
        """clear_cache method documentation."""
        removed = 0
        if model_path is not None:
            key = self._cache_key(model_path, backend)
            if key in self._cache_mem:
                del self._cache_mem[key]
            try:
                path = self._cache_path(model_path, backend)
                if path.exists():
                    path.unlink()
                    removed += 1
            except Exception:
                pass
            return removed

        # Clear all caches
        self._cache_mem.clear()
        cache_dir = self._cache_dir()
        if not cache_dir.exists():
            return removed
        try:
            for p in cache_dir.glob("*.json"):
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
        except Exception:
            pass
        return removed

    def get_cached_pred(
        self, model_path: Path, image_path: Path, backend: str = "yolo"
    ) -> Optional[List[Tuple[float, float, float]]]:
        """get_cached_pred method documentation."""
        preds = self.load_cache(model_path, backend)
        key = str(Path(image_path))
        if key in preds:
            return preds[key]
        key = str(Path(image_path).resolve())
        return preds.get(key)

    def get_cache_for_paths(
        self, model_path: Path, paths: List[Path], backend: str = "yolo"
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        """get_cache_for_paths method documentation."""
        preds = self.load_cache(model_path, backend)
        for p in paths:
            if str(p) not in preds and str(p.resolve()) not in preds:
                return None
        return preds

    def predict(self: object, model_path: Path, image_paths: List[Path], device: str, imgsz: int, conf: float, batch: int, progress_cb: object = None, cancel_cb: object = None, backend: str = 'yolo', sleap_env: Optional[str] = None, sleap_device: str = 'auto', sleap_batch: int = 4, sleap_max_instances: int = 1) -> Tuple[Optional[Dict[str, List[Tuple[float, float, float]]]], str]:
        """predict method documentation."""
        if not image_paths:
            return {}, ""
        if cancel_cb and cancel_cb():
            return None, "Canceled."

        backend = (backend or "yolo").lower()
        if backend == "sleap":
            if not model_path.exists() or not model_path.is_dir():
                return None, f"SLEAP model dir not found: {model_path}"
            if not sleap_env:
                return None, "Select a SLEAP conda env."
            # PoseKit is single-instance only.
            sleap_max_instances = 1
            tmp_dir = self.out_root / "posekit" / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_json = tmp_dir / f"sleap_pred_{os.getpid()}_{uuid.uuid4().hex}.json"
            ok, preds, err = _run_sleap_predict_service(
                model_dir=model_path,
                image_paths=image_paths,
                out_json=out_json,
                keypoint_names=self.keypoint_names,
                skeleton_edges=self.skeleton_edges,
                env_name=sleap_env,
                device=sleap_device,
                batch=sleap_batch,
                max_instances=sleap_max_instances,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            if not ok:
                return None, err
            self.merge_cache(model_path, preds, backend=backend)
            return preds, ""

        if not model_path.exists() or not model_path.is_file():
            return None, f"Weights not found: {model_path}"
        if model_path.suffix != ".pt":
            return None, f"Invalid weights file: {model_path}"

        tmp_dir = self.out_root / "posekit" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_json = tmp_dir / f"pose_pred_{os.getpid()}_{uuid.uuid4().hex}.json"

        ok, preds, err = _run_pose_predict_subprocess(
            weights_path=model_path,
            image_paths=image_paths,
            out_json=out_json,
            device=device,
            imgsz=imgsz,
            conf=conf,
            batch=batch,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        if not ok:
            return None, err

        self.merge_cache(model_path, preds, backend=backend)
        return preds, ""

    @classmethod
    def shutdown_sleap_service(cls: object) -> object:
        """shutdown_sleap_service method documentation."""
        _get_sleap_service().stop()

    @classmethod
    def start_sleap_service(
        cls, env_name: str, out_root: Path
    ) -> Tuple[bool, str, Optional[Path]]:
        """start_sleap_service method documentation."""
        log_path = None
        try:
            log_dir = Path(out_root) / "posekit" / "logs"
            log_path = log_dir / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.log"
        except Exception:
            log_path = None
        ok, err = _get_sleap_service().start(env_name, log_path=log_path)
        return ok, err, log_path

    @classmethod
    def sleap_service_running(cls) -> bool:
        """sleap_service_running method documentation."""
        svc = _get_sleap_service()
        return bool(svc.proc and svc.proc.poll() is None and svc.port)


def _run_pose_predict_subprocess(
    weights_path: Path,
    image_paths: List[Path],
    out_json: Path,
    device: str = "auto",
    imgsz: int = 640,
    conf: float = 0.25,
    batch: int = 16,
    progress_cb=None,
    cancel_cb=None,
) -> Tuple[bool, Dict[str, List[Tuple[float, float, float]]], str]:
    req = {
        "weights": str(weights_path),
        "images": [str(p) for p in image_paths],
        "device": device,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "batch": int(batch),
        "out_json": str(out_json),
    }
    req_path = (
        Path(tempfile.gettempdir())
        / f"pose_pred_req_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    req_path.write_text(json.dumps(req), encoding="utf-8")

    code = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "def _is_cuda_device(d):\n"
        "    d=(d or '').strip().lower()\n"
        "    return d in {'cuda','gpu'} or d.startswith('cuda:') or d.isdigit()\n"
        "def _maybe_limit_cuda_memory():\n"
        "    try:\n"
        "        import torch\n"
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.set_per_process_memory_fraction(0.9)\n"
        "    except Exception:\n"
        "        pass\n"
        "req=Path(sys.argv[1])\n"
        "cfg=json.loads(req.read_text(encoding='utf-8'))\n"
        "from ultralytics import YOLO\n"
        "model=YOLO(cfg['weights'])\n"
        "pred_kwargs={\n"
        "  'imgsz': int(cfg['imgsz']),\n"
        "  'conf': float(cfg['conf']),\n"
        "  'max_det': 1,\n"
        "  'stream': True,\n"
        "  'verbose': False,\n"
        "  'batch': int(cfg['batch']),\n"
        "}\n"
        "if cfg.get('device') and cfg.get('device')!='auto':\n"
        "  pred_kwargs['device']=cfg['device']\n"
        "if _is_cuda_device(cfg.get('device')):\n"
        "  _maybe_limit_cuda_memory()\n"
        "  pred_kwargs['half']=True\n"
        "images=cfg['images']\n"
        "preds={}\n"
        "b=max(1,int(cfg.get('batch',1)))\n"
        "i=0\n"
        "total=len(images)\n"
        "done_total=0\n"
        "while i < len(images):\n"
        "  chunk=images[i:i+b]\n"
        "  kw=dict(pred_kwargs)\n"
        "  kw['source']=chunk\n"
        "  kw['batch']=min(b,len(chunk))\n"
        "  try:\n"
        "    results=list(model.predict(**kw))\n"
        "  except RuntimeError as e:\n"
        "    if 'out of memory' in str(e).lower() and b>1:\n"
        "      b=max(1,b//2)\n"
        "      continue\n"
        "    raise\n"
        "  for ri, r in enumerate(results):\n"
        "    path = None\n"
        "    if r is not None and getattr(r, 'path', None):\n"
        "      path = r.path\n"
        "    elif ri < len(chunk):\n"
        "      path = chunk[ri]\n"
        "    if path is None:\n"
        "      continue\n"
        "    if r is None or r.keypoints is None:\n"
        "      preds[str(Path(path))]=[]\n"
        "      done_total+=1\n"
        "      print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "      continue\n"
        "    k=r.keypoints\n"
        "    try:\n"
        "      xy=k.xy\n"
        "      conf=getattr(k,'conf',None)\n"
        "      xy=xy.cpu().numpy() if hasattr(xy,'cpu') else np.array(xy)\n"
        "      if conf is not None:\n"
        "        conf=conf.cpu().numpy() if hasattr(conf,'cpu') else np.array(conf)\n"
        "    except Exception:\n"
        "      preds[str(Path(path))]=[]\n"
        "      done_total+=1\n"
        "      print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "      continue\n"
        "    if xy.ndim==2:\n"
        "      xy=xy[None,:,:]\n"
        "    if conf is not None and conf.ndim==1:\n"
        "      conf=conf[None,:]\n"
        "    if xy.size==0:\n"
        "      preds[str(Path(path))]=[]\n"
        "      done_total+=1\n"
        "      print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "      continue\n"
        "    if conf is not None:\n"
        "      scores=np.nanmean(conf,axis=1)\n"
        "    else:\n"
        "      scores=None\n"
        "      try:\n"
        "        if r.boxes is not None and hasattr(r.boxes,'conf'):\n"
        "          scores=r.boxes.conf.cpu().numpy()\n"
        "      except Exception:\n"
        "        scores=None\n"
        "      if scores is None:\n"
        "        scores=np.zeros((xy.shape[0],),dtype=np.float32)\n"
        "    best=int(np.argmax(scores)) if len(scores)>0 else 0\n"
        "    pred_xy=xy[best]\n"
        "    pred_conf=conf[best] if conf is not None else np.zeros((pred_xy.shape[0],),dtype=np.float32)\n"
        "    pts=[]\n"
        "    for j in range(pred_xy.shape[0]):\n"
        "      c=float(pred_conf[j]) if j < len(pred_conf) else 0.0\n"
        "      pts.append((float(pred_xy[j][0]),float(pred_xy[j][1]),c))\n"
        "    preds[str(Path(path))]=pts\n"
        "    done_total+=1\n"
        "    print(f'PROGRESS {done_total} {total}', flush=True)\n"
        "  i+=len(chunk)\n"
        "out={'preds':preds}\n"
        "Path(cfg['out_json']).write_text(json.dumps(out),encoding='utf-8')\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(req_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    last_msg = ""
    for line in proc.stdout:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            return False, {}, "Canceled."
        line = line.strip()
        if line.startswith("PROGRESS "):
            try:
                _, done_s, total_s = line.split()
                if progress_cb:
                    progress_cb(int(done_s), int(total_s))
            except Exception:
                pass
        else:
            last_msg = line
    rc = proc.wait()
    if rc != 0:
        return False, {}, (last_msg or "Subprocess failed.")
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return True, data.get("preds", {}), ""
    except Exception as e:
        return False, {}, str(e)


_SLEAP_SERVICE_CODE = textwrap.dedent(
    r"""
import json,sys,threading,traceback,shutil,inspect,gc,subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np

try:
    import sleap_io as sio
except Exception as e:
    print(f'ERROR: sleap_io not installed: {e}', flush=True)
    sys.exit(2)

_state = {'model_dir': None, 'device': None, 'predictor': None}
_log_path = None

def _log(msg):
    try:
        if _log_path:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _normalize_device(device):
    d = (device or "").strip().lower()
    if d == "mps":
        try:
            import torch
            mps_backend = getattr(torch.backends, "mps", None)
            is_built = mps_backend.is_built() if mps_backend and hasattr(mps_backend, "is_built") else None
            is_avail = mps_backend.is_available() if mps_backend and hasattr(mps_backend, "is_available") else None
            _log(f"device mps requested (built={is_built} available={is_avail})")
        except Exception as e:
            _log(f"device mps requested (torch check failed: {e})")
        return "mps"
    return device

def _clear_predictor():
    _state['predictor']=None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _make_skeleton(names, edges):
    try:
        return sio.Skeleton(nodes=names, edges=edges, name='PoseKit')
    except Exception:
        try:
            return sio.Skeleton(nodes=names, edges=edges)
        except Exception:
            if hasattr(sio,'Node') and hasattr(sio,'Skeleton'):
                nodes=[sio.Node(n) for n in names]
                sk=sio.Skeleton(nodes=nodes, name='PoseKit')
                for a,b in edges:
                    try:
                        sk.add_edge(nodes[int(a)], nodes[int(b)])
                    except Exception:
                        pass
                return sk
            raise

def _make_video(paths):
    if hasattr(sio,'load_video'):
        try:
            return sio.load_video(paths)
        except Exception:
            pass
    if hasattr(sio,'Video'):
        V=sio.Video
        if hasattr(V,'from_image_files'):
            return V.from_image_files(paths)
        if hasattr(V,'from_filename') and len(paths)==1:
            return V.from_filename(paths[0])
    raise RuntimeError('Unable to create Video from image list')

def _make_labeled_frames(video, n):
    frames = []
    try:
        LF = getattr(sio, "LabeledFrame", None)
        if LF is None:
            _log("LabeledFrame not found in sleap_io")
            return frames
        for i in range(n):
            try:
                frames.append(LF(video=video, frame_idx=i, instances=[]))
            except Exception:
                frames.append(LF(video=video, frame_idx=i))
    except Exception:
        _log("LabeledFrame creation failed")
        return frames
    return frames

def _load_predictor(model_dir, device):
    if _state.get('predictor') and _state.get('model_dir')==model_dir and _state.get('device')==device:
        return _state.get('predictor')
    _clear_predictor()
    pred = None
    try:
        from sleap_nn.inference import predictors as _pred_mod
        if hasattr(_pred_mod, 'Predictor') and hasattr(_pred_mod.Predictor, 'from_trained_models'):
            pred = _pred_mod.Predictor.from_trained_models(model_paths=[model_dir], device=device)
    except Exception:
        pred = None
    if pred is None:
        try:
            import sleap_nn.predictors as _pred_mod2
            if hasattr(_pred_mod2, 'Predictor') and hasattr(_pred_mod2.Predictor, 'from_trained_models'):
                pred = _pred_mod2.Predictor.from_trained_models(model_paths=[model_dir], device=device)
        except Exception:
            pred = None
    if pred is not None:
        _state['predictor']=pred
        _state['model_dir']=model_dir
        _state['device']=device
        _log(f"predictor: loaded for {model_dir}")
    return pred

def _predict_with_predictor(pred, labels, batch, max_instances):
    for meth in ('predict','predict_labels','run'):
        if hasattr(pred, meth):
            fn=getattr(pred,meth)
            try:
                return fn(labels, batch_size=batch, max_instances=max_instances)
            except TypeError:
                try:
                    return fn(labels)
                except Exception:
                    pass
    return None

def _run_inference(labels, model_dir, device, batch, max_instances):
    device = _normalize_device(device)
    pred = _load_predictor(model_dir, device)
    if pred is not None:
        _log("predictor: using python API")
        try:
            out = _predict_with_predictor(pred, labels, batch, max_instances)
            if out is not None:
                return out
        except Exception as e:
            _log(f"predictor: python API failed: {e}")
            if device == "mps":
                _log("predictor: retrying on cpu")
                return _run_inference(labels, model_dir, "cpu", batch, max_instances)
    try:
        from sleap_nn import predict as _predict_mod
        if hasattr(_predict_mod, 'run_inference'):
            fn=_predict_mod.run_inference
            _log("predictor: using sleap_nn.predict.run_inference")
            sig=inspect.signature(fn)
            kwargs={}
            if 'input_labels' in sig.parameters:
                kwargs['input_labels']=labels
            if 'model_paths' in sig.parameters:
                kwargs['model_paths']=[model_dir]
            if 'device' in sig.parameters and device and device!='auto':
                kwargs['device']=device
            if 'batch_size' in sig.parameters:
                kwargs['batch_size']=batch
            if 'max_instances' in sig.parameters:
                kwargs['max_instances']=max_instances
            try:
                return fn(**kwargs)
            except Exception as e:
                _log(f"predictor: run_inference failed: {e}")
                if device == "mps":
                    _log("predictor: retrying on cpu")
                    return _run_inference(labels, model_dir, "cpu", batch, max_instances)
    except Exception:
        pass
    return None

def _predict_via_cli(cfg, labels):
    tmp_dir = Path(cfg['tmp_dir'])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_slp = tmp_dir / 'input.slp'
    pred_slp = tmp_dir / 'pred.slp'
    if hasattr(sio,'save_file'):
        sio.save_file(labels, input_slp, format='slp')
    else:
        sio.save_slp(labels, input_slp)
    if shutil.which('sleap-nn') is None:
        raise RuntimeError('sleap-nn not found in PATH')
    cmd=['sleap-nn','track','--data_path',str(input_slp),'--model_paths',cfg['model_dir'],
         '--output_path',str(pred_slp)]
    device = _normalize_device(cfg.get('device'))
    if device and device!='auto':
        cmd += ['--device', str(device)]
    if cfg.get('batch'):
        cmd += ['--batch_size', str(int(cfg['batch']))]
    if cfg.get('max_instances'):
        cmd += ['--max_instances', str(int(cfg['max_instances']))]
    res=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode!=0:
        msg=(res.stdout or '').strip() or 'sleap-nn track failed'
        if device == "mps":
            _log(f"sleap-nn: track failed on mps, retrying cpu: {msg}")
            cmd = ['sleap-nn','track','--data_path',str(input_slp),'--model_paths',cfg['model_dir'],
                   '--output_path',str(pred_slp),'--device','cpu']
            if cfg.get('batch'):
                cmd += ['--batch_size', str(int(cfg['batch']))]
            if cfg.get('max_instances'):
                cmd += ['--max_instances', str(int(cfg['max_instances']))]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if res.returncode==0:
                _log('sleap-nn: track done (cpu fallback)')
                return sio.load_file(pred_slp)
            msg=(res.stdout or '').strip() or 'sleap-nn track failed'
        raise RuntimeError(msg)
    _log('sleap-nn: track done')
    return sio.load_file(pred_slp)

def _labels_from_output(out):
    if out is None:
        return None
    if hasattr(out,'labeled_frames') or hasattr(out,'predicted_frames') or hasattr(out,'frames'):
        return out
    if isinstance(out,(list,tuple)):
        for item in out:
            if hasattr(item,'labeled_frames') or hasattr(item,'predicted_frames') or hasattr(item,'frames'):
                return item
    return None

def _labels_summary(labels):
    if labels is None:
        return "labels=None"
    parts=[f"type={type(labels)}"]
    for name in ("labeled_frames","predicted_frames","frames"):
        if hasattr(labels, name):
            try:
                parts.append(f"{name}={len(getattr(labels,name) or [])}")
            except Exception:
                parts.append(f"{name}=?")
    if hasattr(labels, "instances"):
        try:
            parts.append(f"instances={len(getattr(labels,'instances') or [])}")
        except Exception:
            parts.append("instances=?")
    return " ".join(parts)

def _frame_attrs(lf):
    if lf is None:
        return "frame=None"
    parts=[]
    for name in ("predicted_instances","instances","predicted_instance","prediction","predictions","tracks"):
        if hasattr(lf, name):
            try:
                v=getattr(lf, name)
                if isinstance(v, (list,tuple)):
                    parts.append(f"{name}={len(v)}")
                elif v is None:
                    parts.append(f"{name}=None")
                else:
                    parts.append(f"{name}=1")
            except Exception:
                parts.append(f"{name}=?")
    return " ".join(parts) if parts else "no_instance_attrs"

def _labels_first_frame(labels):
    for name in ("labeled_frames","predicted_frames","frames"):
        if hasattr(labels, name):
            try:
                frames = getattr(labels, name) or []
                if frames:
                    return frames[0]
            except Exception:
                pass
    try:
        return labels[0]
    except Exception:
        return None

def _frame_index(lf, default_idx):
    for key in ('frame_idx','frame_index','idx'):
        if hasattr(lf, key):
            try:
                return int(getattr(lf, key))
            except Exception:
                pass
    return int(default_idx)

def _extract_preds(labels, images, num_kpts):
    frames = getattr(labels,'labeled_frames',[]) or []
    if not frames:
        frames = getattr(labels,'predicted_frames',[]) or []
    if not frames:
        frames = getattr(labels,'frames',[]) or []
    if not frames and hasattr(labels,'__len__') and hasattr(labels,'__getitem__'):
        tmp=[]
        for i in range(len(images)):
            try:
                tmp.append(labels[i])
            except Exception:
                pass
        frames = tmp
    by_idx={}
    for idx, lf in enumerate(frames):
        try:
            by_idx[_frame_index(lf, idx)]=lf
        except Exception:
            pass
    preds={}
    for i, path in enumerate(images):
        lf=by_idx.get(i)
        insts=[]
        if lf is not None:
            if getattr(lf,'predicted_instances',None) is not None:
                insts=list(lf.predicted_instances)
            elif getattr(lf,'instances',None) is not None:
                insts=list(lf.instances)
            elif getattr(lf,'predicted_instance',None) is not None:
                insts=[lf.predicted_instance]
            elif getattr(lf,'predictions',None) is not None:
                insts=list(lf.predictions)
            elif getattr(lf,'prediction',None) is not None:
                insts=[lf.prediction]
        if not insts:
            preds[str(Path(path))]=[]
            continue
        best=None
        best_score=-1e9
        for inst in insts:
            score=None
            if hasattr(inst,'score') and inst.score is not None:
                try:
                    score=float(inst.score)
                except Exception:
                    score=None
            if score is None:
                try:
                    arr=inst.numpy(scores=True)
                    if arr is not None and arr.shape[1] >= 3:
                        score=float(np.nanmean(arr[:,2]))
                except Exception:
                    score=None
            if score is None:
                score=0.0
            if best is None or score > best_score:
                best=inst
                best_score=score
        pts=[]
        if best is not None:
            arr=None
            try:
                arr=best.numpy(scores=True)
            except Exception:
                arr=None
            if arr is None:
                try:
                    arr=best.numpy()
                except Exception:
                    arr=None
            if arr is None:
                pts=[(0.0,0.0,0.0) for _ in range(num_kpts)]
            else:
                if arr.shape[1] == 2:
                    arr=np.concatenate([arr, np.zeros((arr.shape[0],1))], axis=1)
                if arr.shape[0] < num_kpts:
                    pad=np.zeros((num_kpts-arr.shape[0],3))
                    arr=np.concatenate([arr, pad], axis=0)
                if arr.shape[0] > num_kpts:
                    arr=arr[:num_kpts]
                for row in arr:
                    x,y,c = float(row[0]), float(row[1]), float(row[2])
                    if not np.isfinite(x) or not np.isfinite(y):
                        pts.append((0.0,0.0,0.0))
                    else:
                        if not np.isfinite(c):
                            c=1.0
                        pts.append((x,y,c))
        preds[str(Path(path))]=pts
    if not any(len(v)>0 for v in preds.values()):
        inst_list = getattr(labels, 'instances', None)
        if inst_list is not None:
            by_idx = {}
            for inst in list(inst_list):
                idx = None
                for key in ('frame_idx','frame_index','idx'):
                    if hasattr(inst, key):
                        try:
                            idx = int(getattr(inst, key))
                            break
                        except Exception:
                            pass
                if idx is None:
                    continue
                by_idx.setdefault(idx, []).append(inst)
            preds = {}
            for i, path in enumerate(images):
                insts = by_idx.get(i, [])
                if not insts:
                    preds[str(Path(path))]=[]
                    continue
                best = insts[0]
                best_score = -1e9
                for inst in insts:
                    score = getattr(inst, 'score', None)
                    try:
                        score = float(score) if score is not None else 0.0
                    except Exception:
                        score = 0.0
                    if score > best_score:
                        best = inst
                        best_score = score
                pts=[]
                try:
                    arr=best.numpy(scores=True)
                except Exception:
                    arr=None
                if arr is None:
                    pts=[(0.0,0.0,0.0) for _ in range(num_kpts)]
                else:
                    if arr.shape[1] == 2:
                        arr=np.concatenate([arr, np.zeros((arr.shape[0],1))], axis=1)
                    if arr.shape[0] < num_kpts:
                        pad=np.zeros((num_kpts-arr.shape[0],3))
                        arr=np.concatenate([arr, pad], axis=0)
                    if arr.shape[0] > num_kpts:
                        arr=arr[:num_kpts]
                    for row in arr:
                        x,y,c = float(row[0]), float(row[1]), float(row[2])
                        if not np.isfinite(x) or not np.isfinite(y):
                            pts.append((0.0,0.0,0.0))
                        else:
                            if not np.isfinite(c):
                                c=1.0
                            pts.append((x,y,c))
                preds[str(Path(path))]=pts
    return preds

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body=json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith('/health'):
            self._json(200, {'ok': True, 'model_dir': _state.get('model_dir')})
        else:
            self._json(404, {'ok': False, 'error': 'Not found'})
    def do_POST(self):
        if self.path.startswith('/shutdown'):
            self._json(200, {'ok': True})
            def _stop():
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            _stop()
            return
        if not self.path.startswith('/infer'):
            self._json(404, {'ok': False, 'error': 'Not found'})
            return
        try:
            length=int(self.headers.get('Content-Length','0'))
            data=self.rfile.read(length).decode('utf-8') if length>0 else '{}'
            cfg=json.loads(data)
            try:
                _log(f"INFER {len(cfg.get('images') or [])} model={cfg.get('model_dir')} device={cfg.get('device')}")
            except Exception:
                pass
            images=cfg.get('images') or []
            names=cfg.get('keypoint_names') or []
            edges=cfg.get('skeleton_edges') or []
            cfg['device'] = _normalize_device(cfg.get('device'))
            sk=_make_skeleton(names, edges)
            video=_make_video(images)
            try:
                _log(f"video type={type(video)}")
            except Exception:
                pass
            frames = _make_labeled_frames(video, len(images))
            try:
                _log(f"frames_created={len(frames)}")
            except Exception:
                pass
            try:
                labels=sio.Labels(videos=[video], skeletons=[sk], labeled_frames=frames)
            except Exception:
                labels=sio.Labels(videos=[video], skeletons=[sk])
            out=_run_inference(labels, cfg.get('model_dir'), cfg.get('device'), int(cfg.get('batch',4)), int(cfg.get('max_instances',1)))
            labels_out=_labels_from_output(out)
            used_cli=False
            if labels_out is None:
                labels_out=_predict_via_cli(cfg, labels)
                used_cli=True
            _log(_labels_summary(labels_out))
            _log(f"frame0: {_frame_attrs(_labels_first_frame(labels_out))}")
            preds=_extract_preds(labels_out, images, len(names))
            if not preds or not any(len(v)>0 for v in preds.values()):
                if not used_cli:
                    labels_out=_predict_via_cli(cfg, labels)
                    _log("fallback: cli used")
                    _log(_labels_summary(labels_out))
                    _log(f"frame0: {_frame_attrs(_labels_first_frame(labels_out))}")
                    preds=_extract_preds(labels_out, images, len(names))
            try:
                nonempty=sum(1 for v in preds.values() if v)
                _log(f"preds_nonempty={nonempty}/{len(images)}")
            except Exception:
                pass
            self._json(200, {'ok': True, 'preds': preds})
        except Exception as e:
            _log(traceback.format_exc())
            self._json(500, {'ok': False, 'error': str(e)})
        finally:
            gc.collect()
    def log_message(self, format, *args):
        return

def main():
    cfg=Path(sys.argv[1])
    data=json.loads(cfg.read_text(encoding='utf-8'))
    global _log_path
    _log_path = data.get('log_path')
    host=data.get('host','127.0.0.1')
    port=int(data.get('port',0))
    httpd=HTTPServer((host, port), Handler)
    actual_port=httpd.server_address[1]
    _log(f'READY {host}:{actual_port}')
    httpd.serve_forever()

if __name__=='__main__':
    main()
"""
).strip()


class _SleapHttpService:
    def __init__(self):
        self.env_name: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._last_log: str = ""
        self.log_path: Optional[Path] = None

    def _drain_stdout(self):
        if not self.proc or not self.proc.stdout:
            return
        log_file = None
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(self.log_path, "a", encoding="utf-8")
            except Exception:
                log_file = None
        for line in self.proc.stdout:
            if line:
                self._last_log = line.strip()
                if log_file:
                    try:
                        log_file.write(line)
                        log_file.flush()
                    except Exception:
                        pass
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass

    def _pick_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def start(self, env_name: str, log_path: Optional[Path] = None) -> Tuple[bool, str]:
        """start method documentation."""
        if self.proc and self.proc.poll() is None and self.env_name == env_name:
            return True, ""
        self.stop()
        self.log_path = log_path
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"[start] env={env_name}\n")
            except Exception:
                pass
        if shutil.which("conda") is None:
            return False, "Conda not found on PATH."
        port = self._pick_free_port()
        cfg = {"host": "127.0.0.1", "port": int(port)}
        if self.log_path:
            cfg["log_path"] = str(self.log_path)
        cfg_path = (
            Path(tempfile.gettempdir())
            / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.json"
        )
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        code_path = (
            Path(tempfile.gettempdir())
            / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.py"
        )
        try:
            code_path.write_text(_SLEAP_SERVICE_CODE, encoding="utf-8")
        except Exception:
            return False, "Failed to write SLEAP service code."
        self.proc = subprocess.Popen(
            ["conda", "run", "-n", env_name, "python", "-u", str(code_path), str(cfg_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        start = time.time()
        self.port = port
        self.env_name = env_name
        while time.time() - start < 15:
            if self.proc.poll() is not None:
                err = self._last_log or "SLEAP service failed to start."
                if self.log_path:
                    err = f"{err} (log: {self.log_path})"
                return False, err
            try:
                url = f"http://127.0.0.1:{self.port}/health"
                with urllib.request.urlopen(url, timeout=0.5) as resp:
                    if resp.status == 200:
                        if self.log_path:
                            try:
                                with open(self.log_path, "a", encoding="utf-8") as f:
                                    f.write("[health] ok\n")
                            except Exception:
                                pass
                        return True, ""
            except Exception:
                time.sleep(0.2)
        self.stop()
        err = "Timed out starting SLEAP service."
        if self.log_path:
            err = f"{err} (log: {self.log_path})"
        return False, err

    def stop(self: object) -> object:
        """stop method documentation."""
        if self.proc and self.proc.poll() is None and self.port:
            try:
                self.request("/shutdown", {}, timeout=2)
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self.port = None
        self.env_name = None

    def request(self, path: str, payload: dict, timeout: float = 3600.0) -> dict:
        """request method documentation."""
        if not self.port:
            raise RuntimeError("SLEAP service not running.")
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}


_SLEAP_SERVICE = _SleapHttpService()


def _get_sleap_service() -> _SleapHttpService:
    return _SLEAP_SERVICE


def _run_sleap_predict_service(
    model_dir: Path,
    image_paths: List[Path],
    out_json: Path,
    keypoint_names: List[str],
    skeleton_edges: List[Tuple[int, int]],
    env_name: str,
    device: str = "auto",
    batch: int = 4,
    max_instances: int = 1,
    progress_cb=None,
    cancel_cb=None,
) -> Tuple[bool, Dict[str, List[Tuple[float, float, float]]], str]:
    svc = _get_sleap_service()
    log_path = None
    try:
        out_root = out_json.parents[2]
        log_dir = out_root / "posekit" / "logs"
        log_path = log_dir / f"sleap_service_{os.getpid()}_{uuid.uuid4().hex}.log"
    except Exception:
        log_path = None
    ok, err = svc.start(env_name, log_path=log_path)
    if not ok:
        return False, {}, err

    total = len(image_paths)
    preds: Dict[str, List[Tuple[float, float, float]]] = {}
    chunk_size = max(1, min(total, max(8, int(batch) * 8)))
    done = 0
    for i in range(0, total, chunk_size):
        if cancel_cb and cancel_cb():
            return False, {}, "Canceled."
        chunk = image_paths[i : i + chunk_size]
        payload = {
            "images": [str(p) for p in chunk],
            "model_dir": str(model_dir),
            "device": device,
            "batch": int(batch),
            "max_instances": int(max_instances),
            "keypoint_names": list(keypoint_names),
            "skeleton_edges": [list(e) for e in skeleton_edges],
            "tmp_dir": str(Path(tempfile.gettempdir()) / f"sleap_srv_{os.getpid()}"),
        }
        try:
            resp = svc.request("/infer", payload, timeout=3600.0)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                err = body or f"HTTP error {e.code}"
                if svc.log_path:
                    err = f"{err} (log: {svc.log_path})"
                return False, {}, err
            except Exception:
                err = f"HTTP error {e.code}"
                if svc.log_path:
                    err = f"{err} (log: {svc.log_path})"
                return False, {}, err
        except Exception as e:
            err = str(e)
            if svc.log_path:
                err = f"{err} (log: {svc.log_path})"
            return False, {}, err
        if not resp or not resp.get("ok"):
            err = resp.get("error", "SLEAP service error.")
            if svc.log_path:
                err = f"{err} (log: {svc.log_path})"
            return False, {}, err
        preds.update(resp.get("preds", {}))
        done += len(chunk)
        if progress_cb:
            progress_cb(done, total)

    out_json.write_text(json.dumps({"preds": preds}), encoding="utf-8")
    return True, preds, ""


def _run_sleap_predict_subprocess(
    model_dir: Path,
    image_paths: List[Path],
    out_json: Path,
    keypoint_names: List[str],
    skeleton_edges: List[Tuple[int, int]],
    env_name: str,
    device: str = "auto",
    batch: int = 4,
    max_instances: int = 1,
    progress_cb=None,
    cancel_cb=None,
) -> Tuple[bool, Dict[str, List[Tuple[float, float, float]]], str]:
    if shutil.which("conda") is None:
        return False, {}, "Conda not found on PATH."

    tmp_dir = Path(tempfile.gettempdir()) / f"sleap_pred_{os.getpid()}_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    req = {
        "model_dir": str(model_dir),
        "images": [str(p) for p in image_paths],
        "out_json": str(out_json),
        "input_slp": str(tmp_dir / "input.slp"),
        "pred_slp": str(tmp_dir / "pred.slp"),
        "keypoint_names": list(keypoint_names),
        "skeleton_edges": [list(e) for e in skeleton_edges],
        "device": device,
        "batch": int(batch),
        "max_instances": int(max_instances),
    }
    req_path = (
        Path(tempfile.gettempdir())
        / f"sleap_pred_req_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    req_path.write_text(json.dumps(req), encoding="utf-8")

    code = (
        "import json,sys,subprocess,shutil,traceback\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "try:\n"
        "    import sleap_io as sio\n"
        "except Exception as e:\n"
        "    print(f'ERROR: sleap_io not installed: {e}')\n"
        "    sys.exit(2)\n"
        "cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "images=cfg.get('images') or []\n"
        "if not images:\n"
        "    Path(cfg['out_json']).write_text(json.dumps({'preds':{}}),encoding='utf-8')\n"
        "    sys.exit(0)\n"
        "def _make_skeleton(names, edges):\n"
        "    try:\n"
        "        return sio.Skeleton(nodes=names, edges=edges, name='PoseKit')\n"
        "    except Exception:\n"
        "        try:\n"
        "            return sio.Skeleton(nodes=names, edges=edges)\n"
        "        except Exception:\n"
        "            if hasattr(sio,'Node') and hasattr(sio,'Skeleton'):\n"
        "                nodes=[sio.Node(n) for n in names]\n"
        "                sk=sio.Skeleton(nodes=nodes, name='PoseKit')\n"
        "                for a,b in edges:\n"
        "                    try:\n"
        "                        sk.add_edge(nodes[int(a)], nodes[int(b)])\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                return sk\n"
        "            raise\n"
        "def _make_video(paths):\n"
        "    if hasattr(sio,'load_video'):\n"
        "        try:\n"
        "            return sio.load_video(paths)\n"
        "        except Exception:\n"
        "            pass\n"
        "    if hasattr(sio,'Video'):\n"
        "        V=sio.Video\n"
        "        if hasattr(V,'from_image_files'):\n"
        "            return V.from_image_files(paths)\n"
        "        if hasattr(V,'from_filename') and len(paths)==1:\n"
        "            return V.from_filename(paths[0])\n"
        "    raise RuntimeError('Unable to create Video from image list')\n"
        "sk=_make_skeleton(cfg.get('keypoint_names') or [], cfg.get('skeleton_edges') or [])\n"
        "video=_make_video(images)\n"
        "try:\n"
        "    labels=sio.Labels(videos=[video], skeletons=[sk], labeled_frames=[])\n"
        "except Exception:\n"
        "    labels=sio.Labels(videos=[video], skeletons=[sk])\n"
        "if hasattr(sio,'save_file'):\n"
        "    sio.save_file(labels, cfg['input_slp'], format='slp')\n"
        "else:\n"
        "    sio.save_slp(labels, cfg['input_slp'])\n"
        "if shutil.which('sleap-nn') is None:\n"
        "    print('ERROR: sleap-nn not found in PATH')\n"
        "    sys.exit(3)\n"
        "cmd=['sleap-nn','track','--data_path',cfg['input_slp'],"
        "'--model_paths',cfg['model_dir'],'--output_path',cfg['pred_slp']]\n"
        "if cfg.get('device') and cfg.get('device')!='auto':\n"
        "    cmd += ['--device', str(cfg['device'])]\n"
        "if cfg.get('batch'):\n"
        "    cmd += ['--batch_size', str(int(cfg['batch']))]\n"
        "if cfg.get('max_instances'):\n"
        "    cmd += ['--max_instances', str(int(cfg['max_instances']))]\n"
        "res=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
        "if res.returncode!=0:\n"
        "    msg=(res.stdout or '').strip() or 'sleap-nn track failed'\n"
        "    print(msg)\n"
        "    sys.exit(res.returncode)\n"
        "labels=sio.load_file(cfg['pred_slp'])\n"
        "labeled=getattr(labels,'labeled_frames',[]) or []\n"
        "by_idx={}\n"
        "for lf in labeled:\n"
        "    try:\n"
        "        by_idx[int(lf.frame_idx)]=lf\n"
        "    except Exception:\n"
        "        pass\n"
        "num_kpts=len(cfg.get('keypoint_names') or [])\n"
        "preds={}\n"
        "total=len(images)\n"
        "for i, path in enumerate(images):\n"
        "    lf=by_idx.get(i)\n"
        "    insts=[]\n"
        "    if lf is not None and getattr(lf,'instances',None) is not None:\n"
        "        insts=list(lf.instances)\n"
        "    if not insts:\n"
        "        preds[str(Path(path))]=[]\n"
        "        print(f'PROGRESS {i+1} {total}', flush=True)\n"
        "        continue\n"
        "    best=None\n"
        "    best_score=-1e9\n"
        "    for inst in insts:\n"
        "        score=None\n"
        "        if hasattr(inst,'score') and inst.score is not None:\n"
        "            try:\n"
        "                score=float(inst.score)\n"
        "            except Exception:\n"
        "                score=None\n"
        "        if score is None:\n"
        "            try:\n"
        "                arr=inst.numpy(scores=True)\n"
        "                if arr is not None and arr.shape[1] >= 3:\n"
        "                    score=float(np.nanmean(arr[:,2]))\n"
        "            except Exception:\n"
        "                score=None\n"
        "        if score is None:\n"
        "            score=0.0\n"
        "        if best is None or score > best_score:\n"
        "            best=inst\n"
        "            best_score=score\n"
        "    pts=[]\n"
        "    if best is not None:\n"
        "        try:\n"
        "            arr=best.numpy(scores=True)\n"
        "        except Exception:\n"
        "            arr=None\n"
        "        if arr is None:\n"
        "            pts=[(0.0,0.0,0.0) for _ in range(num_kpts)]\n"
        "        else:\n"
        "            if arr.shape[1] == 2:\n"
        "                arr=np.concatenate([arr, np.zeros((arr.shape[0],1))], axis=1)\n"
        "            if arr.shape[0] < num_kpts:\n"
        "                pad=np.zeros((num_kpts-arr.shape[0],3))\n"
        "                arr=np.concatenate([arr, pad], axis=0)\n"
        "            if arr.shape[0] > num_kpts:\n"
        "                arr=arr[:num_kpts]\n"
        "            for row in arr:\n"
        "                x,y,c = float(row[0]), float(row[1]), float(row[2])\n"
        "                if not np.isfinite(x) or not np.isfinite(y):\n"
        "                    pts.append((0.0,0.0,0.0))\n"
        "                else:\n"
        "                    if not np.isfinite(c):\n"
        "                        c=0.0\n"
        "                    pts.append((x,y,c))\n"
        "    preds[str(Path(path))]=pts\n"
        "    print(f'PROGRESS {i+1} {total}', flush=True)\n"
        "Path(cfg['out_json']).write_text(json.dumps({'preds':preds}),encoding='utf-8')\n"
    )

    proc = subprocess.Popen(
        ["conda", "run", "-n", env_name, "python", "-c", code, str(req_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    last_msg = ""
    for line in proc.stdout:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            return False, {}, "Canceled."
        line = line.strip()
        if line.startswith("PROGRESS "):
            try:
                _, done_s, total_s = line.split()
                if progress_cb:
                    progress_cb(int(done_s), int(total_s))
            except Exception:
                pass
        elif line:
            last_msg = line
    rc = proc.wait()
    if rc != 0:
        return False, {}, (last_msg or "SLEAP subprocess failed.")
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return True, data.get("preds", {}), ""
    except Exception as e:
        return False, {}, str(e)
