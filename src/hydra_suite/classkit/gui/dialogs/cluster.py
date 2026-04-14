"""ClusterDialog — configure clustering settings for ClassKit."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from hydra_suite.classkit.core.cluster.clustering_backend import (
    probe_clustering_backend,
)


class ClusterDialog(QDialog):
    """Dialog for configuring clustering."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cluster Configuration")
        self.setMinimumWidth(500)

        ann_probe = probe_clustering_backend()
        ann_ready = (
            ann_probe.get("hdbscan_available", False)
            or ann_probe.get("best_ann") != "numpy"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("<h2>Clustering Settings</h2>")
        layout.addWidget(header)

        form = QFormLayout()

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(10, 10000)
        self.n_clusters_spin.setValue(500)
        self.n_clusters_spin.setSuffix(" clusters")
        form.addRow("<b>Number of Clusters:</b>", self.n_clusters_spin)

        self.gpu_combo = QComboBox()
        self.gpu_combo.addItem("CPU (scikit-learn)", False)
        if ann_ready:
            self.gpu_combo.addItem(f"Accelerated ({ann_probe['best_ann']})", True)

        form.addRow("<b>Backend:</b>", self.gpu_combo)

        layout.addLayout(form)

        info_text = (
            "<b>Clustering Tips:</b><br>"
            + "\u2022 Use many clusters (100s-1000s) to capture visual modes<br>"
            + "\u2022 Clusters \u2260 classes (fine-grained structure)<br>"
            + f"\u2022 Current Best Backend: <b>{ann_probe.get('best_ann', 'numpy')}</b>"
        )
        if ann_probe.get("hdbscan_available"):
            info_text += (
                "<br>\u2022 HDBSCAN (density-based) is available for advanced users."
            )

        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #ffffff; line-height: 1.8;"
        )
        layout.addWidget(info)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        return self.n_clusters_spin.value(), self.gpu_combo.currentData()
