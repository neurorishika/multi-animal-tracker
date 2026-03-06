from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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
class Factor:
    name: str
    labels: List[str]
    shortcut_keys: List[str] = field(default_factory=list)


@dataclass
class LabelingScheme:
    name: str
    factors: List[Factor]
    training_modes: List[str]
    description: str = ""

    @property
    def total_classes(self) -> int:
        result = 1
        for f in self.factors:
            result *= len(f.labels)
        return result

    def encode_label(self, factor_values: List[str]) -> str:
        return "|".join(factor_values)

    def decode_label(self, composite: str) -> List[str]:
        return composite.split("|")


@dataclass
class ProjectConfig:
    name: str
    classes: List[str]
    root_dir: Path
    description: str = ""
    scheme: Optional[LabelingScheme] = None


@dataclass
class ClassKitConfig:
    project: ProjectConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    al: ALConfig = field(default_factory=ALConfig)
    viz_umap_n_neighbors: int = 15
    viz_umap_min_dist: float = 0.1
    index_hnsw_m: int = 32
