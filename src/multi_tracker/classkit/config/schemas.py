from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ModelConfig:
    name: str = "vit_b_16"
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class ALConfig:
    batch_size: int = 40
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "uncertainty": 0.4,
            "diversity": 0.35,
            "representativeness": 0.15,
            "audit": 0.10,
        }
    )


@dataclass
class ProjectConfig:
    name: str
    classes: List[str]
    root_dir: Path
    description: str = ""


@dataclass
class ClassKitConfig:
    project: ProjectConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    al: ALConfig = field(default_factory=ALConfig)
    viz_umap_n_neighbors: int = 15
    viz_umap_min_dist: float = 0.1
    index_hnsw_m: int = 32
