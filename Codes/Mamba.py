#!pip install mamba-ssm causal-conv1d tiktoken sentencepiece
#!pip install -q jedi
#!pip -q install --upgrade pip setuptools wheel ninja packaging
#!pip -q install causal-conv1d mamba-ssm    

ok = {}

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_state_update, selective_scan_fn
    )
    ok["selective_state_update"] = selective_state_update is not None
    ok["selective_scan_fn"] = selective_scan_fn is not None
except Exception as e:
    ok["mamba_kernels_error"] = repr(e)

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    ok["causal_conv1d_fn"] = causal_conv1d_fn is not None
    ok["causal_conv1d_update"] = causal_conv1d_update is not None
except Exception as e:
    ok["causal_conv1d_error"] = repr(e)

import json
print(json.dumps(ok, indent=2))


import os
import time
import json
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

# =============================
# Config
# =============================
DATA_PATH = "finance.csv"
TEXT_COL  = "title"
LABEL_COL = "label"

# Mamba only
MODEL_NAME = "state-spaces/mamba-130m-hf"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42
LR = 2e-5
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = torch.cuda.is_available()

# =============================
# Reproducibility
# =============================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================
# Load data
# =============================
data = pd.read_csv(DATA_PATH)
assert TEXT_COL in data.columns and LABEL_COL in data.columns, \
    f"Expected columns {TEXT_COL} and {LABEL_COL} in CSV."

raw_labels = data[LABEL_COL].to_numpy()
classes = np.sort(np.unique(raw_labels))
num_labels = len(classes)

# map labels to 0..K-1 if needed
if not np.array_equal(classes, np.arange(num_labels)):
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in raw_labels], dtype=int)
else:
    y = raw_labels.astype(int)

texts = data[TEXT_COL].astype(str).tolist()

# stratified train/test split (10% test)
train_idx, test_idx = train_test_split(
    np.arange(len(texts)),
    test_size=0.1,
    stratify=y,
    random_state=SEED
)

# =============================
# Tokenizer (Mamba)
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# =============================
# Dataset & Collate
# =============================
class TextLabelDataset(Dataset):
    def __init__(self, texts, labels):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": int(self.labels[idx])}

def collate_fn(examples):
    batch_texts = [ex["text"] for ex in examples]
    batch_labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": batch_labels
    }

full_dataset = TextLabelDataset(texts, y)
train_dataset = Subset(full_dataset, train_idx)
test_dataset  = Subset(full_dataset, test_idx)

# =============================
# Mamba wrapper (masked-mean pooling + linear head)
# =============================
class SeqClsWrapper(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.num_labels = num_labels

        hidden_size = None
        if hasattr(self.backbone.config, "hidden_size"):
            hidden_size = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "d_model"):
            hidden_size = self.backbone.config.d_model
        self.hidden_size = hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels) if hidden_size is not None else None
        self.loss_fn = nn.CrossEntropyLoss()

    def _ensure_classifier(self, last_hidden: torch.Tensor):
        if self.classifier is None:
            self.hidden_size = last_hidden.size(-1)
            self.classifier = nn.Linear(self.hidden_size, self.num_labels).to(last_hidden.device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # [B,T,H]

        # masked mean pooling
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        self._ensure_classifier(last_hidden)
        logits = self.classifier(pooled)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return type("Out", (), {"loss": loss, "logits": logits})


# =============================
# Build model (+ optional gradient checkpointing)
# =============================

def build_model():
    model = SeqClsWrapper(MODEL_NAME, num_labels=num_labels)
    if hasattr(model.backbone, "resize_token_embeddings"):
        model.backbone.resize_token_embeddings(len(tokenizer))
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        try:
            model.backbone.gradient_checkpointing_enable()
        except Exception:
            pass
    return model.to(device)

# =============================
# Train / Eval
# =============================
def train_model(train_loader, val_loader=None, epochs=EPOCHS):
    model = build_model()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() if loss is not None else 0.0

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item() if outputs.loss is not None else 0.0
            val_loss /= max(1, len(val_loader))
            print(f"           Validation Loss: {val_loss:.4f}")

    # sync before timing
    if torch.cuda.is_available(): torch.cuda.synchronize()
    training_time = time.time() - start_time
    return model, optimizer, scheduler, training_time

@torch.no_grad()
def evaluate_model(dataloader, model):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# =============================
# Metrics (matches Code 1)
# =============================
def calculate_standard_error(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) <= 1:
      return np.nan
    return np.std(v, ddof=1) / np.sqrt(len(v))

def calculate_error_measures(true_labels, predictions, probs, num_classes):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    full_labels = np.arange(num_classes)
    cm = confusion_matrix(true_labels, predictions, labels=full_labels)

    tp = np.zeros(num_classes); fp = np.zeros(num_classes)
    fn = np.zeros(num_classes); tn = np.zeros(num_classes)

    fp_sources = {i: {} for i in range(num_classes)}
    fn_sources = {i: {} for i in range(num_classes)}

    for i in range(num_classes):
        tp[i] = cm[i, i]
        fp[i] = np.sum(cm[:, i]) - tp[i]
        fn[i] = np.sum(cm[i, :]) - tp[i]
        tn[i] = np.sum(cm) - (tp[i] + fp[i] + fn[i])
        for j in range(num_classes):
            if j != i and cm[j, i] > 0: fp_sources[i][j] = int(cm[j, i])
            if j != i and cm[i, j] > 0: fn_sources[i][j] = int(cm[i, j])

    true_bin = label_binarize(true_labels, classes=full_labels)
    probs_arr = np.asarray(probs)

    fpr, tpr, roc_auc = {}, {}, {}
    class_metrics = {}

    for i in range(num_classes):
        y = true_bin[:, i]; s = probs_arr[:, i]
        if y.max() != y.min():
            fpr[i], tpr[i], _ = roc_curve(y, s)
            roc_auc[i] = roc_auc_score(y, s)
        else:
            fpr[i], tpr[i], roc_auc[i] = np.array([np.nan]), np.array([np.nan]), np.nan

        denom = tp[i] + tn[i] + fp[i] + fn[i]
        class_metrics[i] = {
            "tp": tp[i], "tn": tn[i], "fp": fp[i], "fn": fn[i],
            "accuracy": (tp[i] + tn[i]) / denom if denom > 0 else 0.0,
            "precision": tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0,
            "recall": tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0,
            "f1_score": (2 * tp[i]) / (2 * tp[i] + fp[i] + fn[i]) if (2 * tp[i] + fp[i] + fn[i]) > 0 else 0.0,
            "fpr": fpr.get(i, None), "tpr": tpr.get(i, None), "auc": roc_auc.get(i, np.nan),
            "fp_sources": fp_sources[i], "fn_sources": fn_sources[i]
        }

    roc_auc_macro = np.nanmean([roc_auc[i] for i in range(num_classes)])
    per_class_fpr_at_op = [
        (fp[i] / (fp[i] + tn[i])) if (fp[i] + tn[i]) > 0 else np.nan
        for i in range(num_classes)
    ]
    fpr_macro = np.nanmean(per_class_fpr_at_op)

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
        "tp": tp.sum(), "tn": tn.sum(), "fp": fp.sum(), "fn": fn.sum(),
        "fpr": fpr_macro, "tpr": tpr,
        "auc": {"macro": roc_auc_macro, **roc_auc},
        "class_metrics": class_metrics, "confusion_matrix": cm
    }

def extract_numeric_metrics(metrics):
    return {
        "accuracy": metrics["accuracy"], "precision": metrics["precision"],
        "recall": metrics["recall"], "f1_score": metrics["f1_score"],
        "tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"],
        "auc": metrics["auc"].get("macro", np.nan), "fpr": metrics["fpr"]
    }

def sources_from_cm(cm):
    num_classes = cm.shape[0]
    fp_sources = {i: {} for i in range(num_classes)}
    fn_sources = {i: {} for i in range(num_classes)}
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j: continue
            if cm[j, i] > 0: fp_sources[i][j] = int(cm[j, i])
            if cm[i, j] > 0: fn_sources[i][j] = int(cm[i, j])
    return fp_sources, fn_sources

# =============================
# Entity helpers (for VALIDATION entity-level metrics at 100%)
# =============================
qualified_entities = ["@PER", "@ORG", "@LOC", "@EVENT", "@PROD", "@CHEM", "@SYMPTOM", "@VACCINE",
                      "@MISC", "@TECH", "@BIO", "@DISEASE", "@PERCENT", "@MONEY", "@DATE", "@TIME",
                      "@LAW", "@TREATMENT"]

def extract_entity(title):
    for tag in qualified_entities:
        if tag in title:
            return tag
    return "No Entity"

# =============================
# Cross-validation over train-size percentages (TRAIN + VAL)
# =============================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_pcts = [3, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

training_rows = []   # like Code 1
train_cm_rows = []   # wide stringified TRAIN & VAL CMs (like Code 1)
val_relation_rows = []  # NEW: entity-level metrics at 100%

for pct in train_pcts:
    n_train_total = len(train_idx)
    subset_size = int(round(n_train_total * (pct / 100)))
    subset_size = max(subset_size, kf.get_n_splits())
    subset_size = min(subset_size, n_train_total)

    train_idx_arr = np.array(train_idx)
    train_labels_all = y[train_idx_arr]

    if subset_size == n_train_total:
        subset_indices = train_idx_arr
        subset_labels = train_labels_all
    else:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=SEED)
        (subset_rel_idx, _), = sss.split(np.arange(n_train_total), train_labels_all)
        subset_indices = train_idx_arr[subset_rel_idx]
        subset_labels = y[subset_indices]

    subset_dataset = Subset(full_dataset, subset_indices)

    times = []

    # TRAIN aggregations
    overall_collect_train = {k: [] for k in ["accuracy","precision","recall","f1_score","tp","tn","fp","fn","auc","fpr"]}
    class_collect_train = {}
    sum_cm_train = np.zeros((num_labels, num_labels), dtype=int)

    # VAL aggregations
    overall_collect_val = {k: [] for k in ["accuracy","precision","recall","f1_score","tp","tn","fp","fn","auc","fpr"]}
    class_collect_val = {}
    sum_cm_val = np.zeros((num_labels, num_labels), dtype=int)

    # NEW: accumulate validation titles/labels/preds at 100%
    val_titles_accum, val_true_accum, val_pred_accum = [], [], []

    for tr_rel_idx, va_rel_idx in kf.split(np.arange(len(subset_dataset)), subset_labels):
        train_subset = Subset(subset_dataset, tr_rel_idx)
        val_subset   = Subset(subset_dataset, va_rel_idx)

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_subset),
            collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, sampler=SequentialSampler(val_subset),
            collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory
        )

        model, optimizer, scheduler, training_time = train_model(train_loader, val_loader, epochs=EPOCHS)
        times.append(training_time)

        # TRAIN metrics
        tr_pred, tr_true, tr_probs = evaluate_model(train_loader, model)
        tr_metrics = calculate_error_measures(tr_true, tr_pred, tr_probs, num_classes=num_labels)
        sum_cm_train += tr_metrics["confusion_matrix"]
        nm_tr = extract_numeric_metrics(tr_metrics)
        for k in overall_collect_train:
            overall_collect_train[k].append(nm_tr[k])
        for cls, cm in tr_metrics["class_metrics"].items():
            if cls not in class_collect_train:
                class_collect_train[cls] = {m: [] for m in ["accuracy","precision","recall","f1_score","tp","tn","fp","fn","auc"]}
            for m in class_collect_train[cls]:
                class_collect_train[cls][m].append(cm.get(m, np.nan))

        # VAL metrics
        va_pred, va_true, va_probs = evaluate_model(val_loader, model)
        va_metrics = calculate_error_measures(va_true, va_pred, va_probs, num_classes=num_labels)
        sum_cm_val += va_metrics["confusion_matrix"]
        nm_val = extract_numeric_metrics(va_metrics)
        for k in overall_collect_val:
            overall_collect_val[k].append(nm_val[k])
        for cls, cm in va_metrics["class_metrics"].items():
            if cls not in class_collect_val:
                class_collect_val[cls] = {m: [] for m in ["accuracy","precision","recall","f1_score","tp","tn","fp","fn","auc"]}
            for m in class_collect_val[cls]:
                class_collect_val[cls][m].append(cm.get(m, np.nan))

        # NEW: collect entity-level data for 100% training size
        if pct == 100:
            # map fold-relative indices to original dataframe rows
            val_orig_indices = [subset_indices[i] for i in va_rel_idx]
            val_titles = data.loc[val_orig_indices, TEXT_COL].astype(str).tolist()
            val_titles_accum.extend(val_titles)
            val_true_accum.extend(va_true.tolist())
            val_pred_accum.extend(va_pred.tolist())

        # free per fold
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # aggregated FP/FN sources (kept in the training_results row as JSON)
    train_fp_sources, train_fn_sources = sources_from_cm(sum_cm_train)
    val_fp_sources,   val_fn_sources   = sources_from_cm(sum_cm_val)

    # mean ± SE helper
    def mean_se(v):
        return float(np.nanmean(v)), float(calculate_standard_error(v))

    row = {
        "percent_training_data": pct,
        "training_time_mean": float(np.nanmean(times)),
        "training_time_se": float(calculate_standard_error(times)),
    }

    # TRAIN means + SE
    for k, vals in overall_collect_train.items():
        m, se = mean_se(vals)
        row[f"train_{k}_mean"] = m
        row[f"train_{k}_se"]   = se

    # VAL means + SE
    for k, vals in overall_collect_val.items():
        m, se = mean_se(vals)
        row[f"val_{k}_mean"] = m
        row[f"val_{k}_se"]   = se

    # class-wise TRAIN & VAL + sources
    for cls in range(num_labels):
        md_tr = class_collect_train.get(cls, None)
        if md_tr is not None:
            for mname, vals in md_tr.items():
                m, se = mean_se(vals)
                row[f"train_class_{cls}_{mname}_mean"] = m
                row[f"train_class_{cls}_{mname}_se"]   = se
        row[f"train_class_{cls}_fp_sources"] = json.dumps(train_fp_sources.get(cls, {}))
        row[f"train_class_{cls}_fn_sources"] = json.dumps(train_fn_sources.get(cls, {}))

    for cls in range(num_labels):
        md_val = class_collect_val.get(cls, None)
        if md_val is not None:
            for mname, vals in md_val.items():
                m, se = mean_se(vals)
                row[f"val_class_{cls}_{mname}_mean"] = m
                row[f"val_class_{cls}_{mname}_se"]   = se
        row[f"val_class_{cls}_fp_sources"] = json.dumps(val_fp_sources.get(cls, {}))
        row[f"val_class_{cls}_fn_sources"] = json.dumps(val_fn_sources.get(cls, {}))

    training_rows.append(row)

    # ---- Print wide CMs and store them (stringified) — like Code 1 ----
    print(f"\n[TRAIN] Confusion Matrix (aggregated over folds) for {pct}% of training data:")
    cm_train_df = pd.DataFrame(
        sum_cm_train,
        index=[f"T{i}" for i in range(num_labels)],
        columns=[f"P{j}" for j in range(num_labels)]
    )
    print(cm_train_df)

    print(f"\n[VAL]   Confusion Matrix (aggregated over folds) for {pct}% of training data:")
    cm_val_df = pd.DataFrame(
        sum_cm_val,
        index=[f"T{i}" for i in range(num_labels)],
        columns=[f"P{j}" for j in range(num_labels)]
    )
    print(cm_val_df)

    train_cm_rows.append({
        "percent_training_data": pct,
        "train_confusion_matrix": cm_train_df.to_string(),
        "val_confusion_matrix":   cm_val_df.to_string()
    })

    # ---- Pretty-print TRAIN / VAL scalar metrics (mean ± SE), incl. counts ----
    def _mk_scalar_df(prefix, rowd):
        metrics = ["accuracy","precision","recall","f1_score","auc","fpr","tp","tn","fp","fn"]
        rows = []
        for m in metrics:
            mean_key = f"{prefix}_{m}_mean"
            se_key   = f"{prefix}_{m}_se"
            if mean_key in rowd:
                rows.append({"metric": m, "mean": rowd[mean_key], "se": rowd.get(se_key, np.nan)})
        return pd.DataFrame(rows)

    train_scalar_df = _mk_scalar_df("train", row)
    val_scalar_df   = _mk_scalar_df("val", row)
    print(f"\n[{pct}%] TRAIN scalar metrics (mean ± SE)")
    print(train_scalar_df.round(4).to_string(index=False))
    print(f"\n[{pct}%] VAL   scalar metrics (mean ± SE)")
    print(val_scalar_df.round(4).to_string(index=False))

    # ---- Pretty-print TRAIN / VAL class-wise metrics (mean ± SE) ----
    def _mk_classwise_df(prefix):
        cols = ["accuracy","precision","recall","f1_score","auc","tp","tn","fp","fn"]
        out = []
        for cls in range(num_labels):
            row_cls = {"class": cls}
            any_present = False
            for m in cols:
                m_mean = row.get(f"{prefix}_class_{cls}_{m}_mean", np.nan)
                m_se   = row.get(f"{prefix}_class_{cls}_{m}_se",   np.nan)
                row_cls[f"{m}_mean"] = m_mean
                row_cls[f"{m}_se"]   = m_se
                if not (isinstance(m_mean, float) and np.isnan(m_mean)):
                    any_present = True
            if any_present:
                out.append(row_cls)
        return pd.DataFrame(out)

    print(f"\n[{pct}%] TRAIN class-wise metrics (mean ± SE)")
    print(_mk_classwise_df("train").round(4).to_string(index=False))
    print(f"\n[{pct}%] VAL   class-wise metrics (mean ± SE)")
    print(_mk_classwise_df("val").round(4).to_string(index=False))

    # ===== NEW: After folds, write entity-level metrics for 100% training size =====
    if pct == 100 and len(val_true_accum) > 0:
        val_df = pd.DataFrame({
            "True Labels": val_true_accum,
            "Predicted Labels": val_pred_accum,
            "title": val_titles_accum
        })
        val_df["Entity"] = val_df["title"].apply(extract_entity)

        # collect per-class × entity metrics
        for class_id in sorted(set(val_true_accum)):
            cdf = val_df[val_df["True Labels"] == class_id]
            for ent in sorted(cdf["Entity"].unique()):
                sub = cdf[cdf["Entity"] == ent]
                if len(sub) == 0:
                    continue
                acc = accuracy_score(sub["True Labels"], sub["Predicted Labels"])
                f1v = f1_score(sub["True Labels"], sub["Predicted Labels"], average="macro", zero_division=0)
                val_relation_rows.append({
                    "class_id": int(class_id),
                    "relation_type": ent,
                    "accuracy": float(acc),
                    "f1_score": float(f1v),
                    "sample_count": int(len(sub))
                })

# ===== After the percentage loop: save TRAIN/VAL CSVs (like Code 1) =====
pd.DataFrame(training_rows).to_csv("training_results.csv", index=False)
pd.DataFrame(train_cm_rows).to_csv("training_confusion_matrices.csv", index=False)
if len(val_relation_rows) > 0:
    pd.DataFrame(val_relation_rows).to_csv("val_relation_metrics.csv", index=False)

print("\n[Saved]")
print(" - training_results.csv            (train & val, means + SE per %)")
print(" - training_confusion_matrices.csv (wide stringified TRAIN & VAL CMs per %)")
if len(val_relation_rows) > 0:
    print(" - val_relation_metrics.csv         (entity-level VAL metrics at 100%)")

# =============================
# Train final model on full train, for test evaluation
# =============================
train_loader_full = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset),
    collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory
)
final_model, final_opt, final_sched, final_train_time = train_model(train_loader_full, None, epochs=EPOCHS)
print(f"\n[FINAL] Full-train time: {final_train_time:.2f}s")
pd.DataFrame([{
    "final_full_train_time_sec": final_train_time,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
}]).to_csv("final_train_time.csv", index=False)

# ==== Evaluate once on FULL test set (OVERALL + PER-CLASS + WIDE CM) ====
test_loader_full = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(test_dataset),
    collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=pin_memory
)
test_preds, test_true, _test_probs = evaluate_model(test_loader_full, final_model)

test_metrics = calculate_error_measures(
    true_labels=test_true,
    predictions=test_preds,
    probs=_test_probs,
    num_classes=num_labels
)

# Build test CSV like Code 1: overall + per-class (no confusion rows)
rows_test_all = []

# overall row
overall_entry = extract_numeric_metrics(test_metrics)
overall_entry["section"] = "overall"
rows_test_all.append(overall_entry)

# per-class rows
for cls, md in sorted(test_metrics["class_metrics"].items()):
    rows_test_all.append({
        "section": "per_class",
        "class": cls,
        "tp": md["tp"],
        "tn": md["tn"],
        "fp": md["fp"],
        "fn": md["fn"],
        "accuracy": md["accuracy"],
        "precision": md["precision"],
        "recall": md["recall"],
        "f1_score": md["f1_score"],
        "auc": md["auc"],
    })

test_all_df = pd.DataFrame(rows_test_all)
test_all_df.to_csv("test_metrics.csv", index=False)

# Print convenience splits (like Code 1)
print("\n[TEST] Overall scalar metrics")
print(test_all_df[test_all_df["section"]=="overall"].drop(columns=["section"]).round(4).to_string(index=False))

print("\n[TEST] Per-class metrics")
print(test_all_df[test_all_df["section"]=="per_class"].drop(columns=["section"]).round(4).to_string(index=False))

# Confusion matrix: print & save wide (like Code 1)
cm_test_df = pd.DataFrame(
    test_metrics["confusion_matrix"],
    index=[f"T{i}" for i in range(num_labels)],
    columns=[f"P{j}" for j in range(num_labels)]
)
print("\n[TEST] Confusion matrix")
print(cm_test_df)
cm_test_df.to_csv("test_confusion_matrix.csv", index=True)

print("\n[TEST] Saved:")
print(" - test_metrics.csv")
print(" - test_confusion_matrix.csv")

# Free final model (optional)
del final_model, final_opt, final_sched
torch.cuda.empty_cache()
gc.collect()

print("\nAll done ✅")
