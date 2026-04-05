"""classkit GUI dialogs package."""

from hydra_suite.classkit.gui.dialogs._helpers import (
    _KeyCapture,
    _LabelRow,
    _SchemeWrapper,
)
from hydra_suite.classkit.gui.dialogs.add_source import AddSourceDialog
from hydra_suite.classkit.gui.dialogs.apriltag_autolabel import AprilTagAutoLabelDialog
from hydra_suite.classkit.gui.dialogs.class_editor import ClassEditorDialog
from hydra_suite.classkit.gui.dialogs.cluster import ClusterDialog
from hydra_suite.classkit.gui.dialogs.embedding import EmbeddingDialog
from hydra_suite.classkit.gui.dialogs.export import ExportDialog
from hydra_suite.classkit.gui.dialogs.model_history import ModelHistoryDialog
from hydra_suite.classkit.gui.dialogs.new_project import NewProjectDialog
from hydra_suite.classkit.gui.dialogs.shortcut_editor import ShortcutEditorDialog
from hydra_suite.classkit.gui.dialogs.source_manager import SourceManagerDialog
from hydra_suite.classkit.gui.dialogs.training import ClassKitTrainingDialog

__all__ = [
    "_KeyCapture",
    "_LabelRow",
    "_SchemeWrapper",
    "AddSourceDialog",
    "SourceManagerDialog",
    "ClassEditorDialog",
    "ShortcutEditorDialog",
    "NewProjectDialog",
    "EmbeddingDialog",
    "ClusterDialog",
    "ClassKitTrainingDialog",
    "ExportDialog",
    "ModelHistoryDialog",
    "AprilTagAutoLabelDialog",
]
