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
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PoseInferenceService:
    CACHE_VERSION = 2

    def __init__(self, out_root: Path, keypoint_names: List[str]):
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        kpt_sig_src = "|".join(self.keypoint_names) + f":{len(self.keypoint_names)}"
        self.kpt_sig = hashlib.sha1(kpt_sig_src.encode("utf-8")).hexdigest()[:12]
        self._cache_mem: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}

    def _weights_sig(self, weights_path: Path) -> Optional[str]:
        if not weights_path.exists() or not weights_path.is_file():
            return None
        stat = weights_path.stat()
        token = f"{weights_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _cache_key(self, weights_path: Path) -> str:
        w = self._weights_sig(weights_path) or "missing"
        token = f"v{self.CACHE_VERSION}|k{self.kpt_sig}|n{len(self.keypoint_names)}|w{w}"
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _cache_dir(self) -> Path:
        return self.out_root / ".posekit" / "predictions"

    def _cache_path(self, weights_path: Path) -> Path:
        return self._cache_dir() / f"{self._cache_key(weights_path)}.json"

    def _read_cache_file(
        self, path: Path
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            if int(meta.get("version", -1)) != self.CACHE_VERSION:
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
        self, weights_path: Path
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        key = self._cache_key(weights_path)
        if key in self._cache_mem:
            return self._cache_mem[key]
        preds = self._read_cache_file(self._cache_path(weights_path)) or {}
        self._cache_mem[key] = preds
        return preds

    def _write_cache(
        self, weights_path: Path, preds: Dict[str, List[Tuple[float, float, float]]]
    ):
        self._cache_dir().mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self.CACHE_VERSION,
            "kpt_sig": self.kpt_sig,
            "num_kpts": len(self.keypoint_names),
            "weights_path": str(weights_path),
            "weights_sig": self._weights_sig(weights_path),
        }
        payload = {"meta": meta, "preds": preds}
        path = self._cache_path(weights_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._cache_mem[self._cache_key(weights_path)] = preds

    def merge_cache(
        self, weights_path: Path, new_preds: Dict[str, List[Tuple[float, float, float]]]
    ):
        cache = self.load_cache(weights_path)
        cache.update(new_preds)
        self._write_cache(weights_path, cache)

    def get_cached_pred(
        self, weights_path: Path, image_path: Path
    ) -> Optional[List[Tuple[float, float, float]]]:
        preds = self.load_cache(weights_path)
        key = str(Path(image_path))
        if key in preds:
            return preds[key]
        key = str(Path(image_path).resolve())
        return preds.get(key)

    def get_cache_for_paths(
        self, weights_path: Path, paths: List[Path]
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        preds = self.load_cache(weights_path)
        for p in paths:
            if str(p) not in preds and str(p.resolve()) not in preds:
                return None
        return preds

    def predict(
        self,
        weights_path: Path,
        image_paths: List[Path],
        device: str,
        imgsz: int,
        conf: float,
        batch: int,
        progress_cb=None,
        cancel_cb=None,
    ) -> Tuple[Optional[Dict[str, List[Tuple[float, float, float]]]], str]:
        if not weights_path.exists() or not weights_path.is_file():
            return None, f"Weights not found: {weights_path}"
        if weights_path.suffix != ".pt":
            return None, f"Invalid weights file: {weights_path}"
        if not image_paths:
            return {}, ""
        if cancel_cb and cancel_cb():
            return None, "Canceled."

        tmp_dir = self.out_root / ".posekit" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_json = tmp_dir / f"pose_pred_{os.getpid()}_{uuid.uuid4().hex}.json"

        ok, preds, err = _run_pose_predict_subprocess(
            weights_path=weights_path,
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

        self.merge_cache(weights_path, preds)
        return preds, ""


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
