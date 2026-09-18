"""MetalCytoToxDB QSAR toxicity dataset.

This module contains :class:`MetalCytoToxDataset`, a tabular QSAR
toxicity loader that reads ``data/MetalCytoToxDB.csv`` (numeric
descriptors + a label column), handles missing values, one-hot encodes
categorical columns, and exposes the result as a list of
:class:`data._base.MoleculeSample` objects.

The dataset lives in :mod:`data.metalcytotox` but is re-exported from
:mod:`data.mol_dataset` and :mod:`data` for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ._base import BaseMoleculeDataset, MoleculeSample

__all__ = ["MetalCytoToxDataset"]


class MetalCytoToxDataset(BaseMoleculeDataset):
    """QSAR toxicity dataset loaded from ``data/MetalCytoToxDB.csv``.

    The CSV mixes numeric descriptors (molecular weight, logP, etc.)
    with categorical descriptors (e.g. metal species).  Missing values
    are filled column-wise:

    * numeric columns -> column mean (computed on the training set
      so we don't leak label info; for an unsupervised dataset this is
      a no-op).
    * categorical / object columns -> the most frequent value.

    Featurization:

    * numeric columns are stacked into a single float32 vector.
    * categorical columns are one-hot encoded using categories seen at
      load time (so unknown categories at eval time become all-zero
      vectors, which is the standard sklearn behaviour).

    The label is read from ``label_column`` (default ``"label"``).
    If no such column exists we look for a column literally named
    ``"y"`` or ``"activity"``, falling back to the last column.

    Parameters
    ----------
    root:
        Directory that contains ``MetalCytoToxDB.csv``.  Defaults to
        ``Path("data")``.
    csv_name:
        Filename inside ``root``.
    label_column:
        Name of the label column.  ``None`` tries common names first
        and finally the last column.
    drop_columns:
        Columns to discard before featurization (e.g. molecule IDs,
        SMILES strings).  ``None`` auto-drops columns whose dtype is
        ``object`` *and* which contain the substring ``"smile"`` or
        ``"mol"`` (case-insensitive) - this keeps obvious string
        identifiers out of the numeric pipeline without forcing the
        caller to maintain a list.
    """

    NUMERIC_DTYPES = {"int64", "int32", "int16", "int8", "float64", "float32"}

    def __init__(
        self,
        root: str | Path | None = None,
        csv_name: str = "MetalCytoToxDB.csv",
        label_column: str | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(root=root)
        self.csv_path = self.root / csv_name
        self.label_column = label_column
        self.drop_columns = list(drop_columns) if drop_columns else []

        # Filled by _load_raw.
        self.feature_dim: int = 0
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []

        self._load_raw()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_raw(self) -> None:
        if not self.csv_path.exists():
            # Empty / not-yet-populated CSV: we don't want a hard
            # crash in CI.  Raise a clear error so callers can stub
            # the path instead.
            raise FileNotFoundError(
                f"MetalCytoToxDB CSV not found at {self.csv_path}. "
                "Either populate the file or pass a different `root` / "
                "`csv_name`."
            )

        df = pd.read_csv(self.csv_path)

        # 1. resolve label column ------------------------------------------------
        label_col = self._resolve_label_column(df)
        labels = df[label_col].to_numpy()
        df = df.drop(columns=[label_col])

        # 2. drop unwanted string-y columns -------------------------------------
        auto_drop = [
            c for c in df.columns
            if df[c].dtype == "object"
            and any(token in c.lower() for token in ("smile", "mol", "name", "id"))
        ]
        for c in auto_drop:
            if c not in self.drop_columns:
                self.drop_columns.append(c)
        df = df.drop(columns=[c for c in self.drop_columns if c in df.columns])

        # 3. missing-value handling ---------------------------------------------
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype.name in self.NUMERIC_DTYPES:
                    fill = df[col].mean()
                else:
                    # categorical -> mode (first value if mode is empty)
                    mode = df[col].mode(dropna=True)
                    fill = mode.iloc[0] if not mode.empty else "missing"
                df[col] = df[col].fillna(fill)

        # 4. split numeric vs categorical ---------------------------------------
        numeric_cols = [
            c for c in df.columns if df[c].dtype.name in self.NUMERIC_DTYPES
        ]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        # 5. numeric -> float32 stack ------------------------------------------
        numeric_features = (
            df[numeric_cols].to_numpy(dtype=np.float32) if numeric_cols else
            np.zeros((len(df), 0), dtype=np.float32)
        )

        # 6. categorical -> one-hot --------------------------------------------
        cat_features = np.zeros((len(df), 0), dtype=np.float32)
        cat_cardinalities: list[int] = []
        if categorical_cols:
            cat_frames = []
            for col in categorical_cols:
                # ``astype('category')`` gives stable, deterministic ordering
                # (sorted categories) which is what we want here.
                series = df[col].astype("category")
                cat_frames.append(
                    np.eye(len(series.cat.categories), dtype=np.float32)[
                        series.cat.codes.to_numpy()
                    ]
                )
                cat_cardinalities.append(len(series.cat.categories))
            cat_features = np.concatenate(cat_frames, axis=1)

        # 7. final feature matrix ----------------------------------------------
        if numeric_features.size and cat_features.size:
            features = np.concatenate([numeric_features, cat_features], axis=1)
        elif numeric_features.size:
            features = numeric_features
        else:
            features = cat_features

        # 8. pack into MoleculeSample list --------------------------------------
        features_t = torch.from_numpy(features)  # (N, F)
        if labels.dtype.kind in {"i", "u", "f"}:
            labels_t = torch.from_numpy(labels.astype(np.float32))
        else:
            # string labels -> class indices; rare path but keep it sane.
            classes, labels_idx = np.unique(labels, return_inverse=True)
            labels_t = torch.from_numpy(labels_idx.astype(np.int64))
            self._label_classes = classes  # type: ignore[attr-defined]

        self._samples = [
            MoleculeSample(features=features_t[i], label=labels_t[i])
            for i in range(len(df))
        ]

        self.feature_dim = int(features_t.shape[-1])
        self.feature_columns = list(numeric_cols)
        self.categorical_columns = list(categorical_cols)

    def _resolve_label_column(self, df: pd.DataFrame) -> str:
        if self.label_column and self.label_column in df.columns:
            return self.label_column
        for candidate in ("label", "y", "activity", "toxicity", "target"):
            if candidate in df.columns:
                return candidate
        # last column fallback
        return df.columns[-1]