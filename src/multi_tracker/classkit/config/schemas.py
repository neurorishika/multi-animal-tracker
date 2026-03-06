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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "labels": self.labels,
            "shortcut_keys": self.shortcut_keys,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Factor":
        return cls(
            name=d["name"],
            labels=d["labels"],
            shortcut_keys=d.get("shortcut_keys", []),
        )


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
        if len(factor_values) != len(self.factors):
            raise ValueError(
                f"Expected {len(self.factors)} factor values, got {len(factor_values)}"
            )
        return "|".join(factor_values)

    def decode_label(self, composite: str) -> List[str]:
        parts = composite.split("|")
        if len(parts) != len(self.factors):
            raise ValueError(
                f"Expected {len(self.factors)} parts in composite label, got {len(parts)}"
            )
        return parts

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "factors": [f.to_dict() for f in self.factors],
            "training_modes": self.training_modes,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelingScheme":
        return cls(
            name=d["name"],
            factors=[Factor.from_dict(f) for f in d.get("factors", [])],
            training_modes=d.get("training_modes", []),
            description=d.get("description", ""),
        )


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
