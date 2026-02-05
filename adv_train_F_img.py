import os, json, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import clip
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    balanced_accuracy_score,
    f1_score, precision_score, recall_score
)

# ----------------------------
# 1. Config
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

csv_path = "/data/rashidm/test code/SCP/labeled_dataset.csv"
image_base_path = "/data/rashidm/car-parts/offerup"
batch_size = 16
epochs = 3
lr = 1e-5
MAX_IMAGES = 1
log_filename = "adv_training_joint_Fimg_text.txt"

# --- Image PGD Config (pixel-space budget) ---
PGD_EPS   = 8 / 255
PGD_ALPHA = 1 / 255
PGD_STEPS = 5

# --- Text PGD Config (embedding-space) ---
TXT_EPS   = 0.05
TXT_ALPHA = 0.01
TXT_STEPS = 5

# Evaluation attack steps (joint)
EVAL_STEPS = 10

# Loss mixing (clean + adv)
CLEAN_WEIGHT = 0.5
ADV_WEIGHT   = 0.5

# ----------------------------
# 2. Load CLIP & (optional) Freeze encoders
# ----------------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float().to(device)

# NOTE:
# - You can freeze encoders if you want to train only fc.
# - Attacks (PGD) will still work because we take gradients w.r.t. inputs (images / delta_txt).
# Keep your previous behavior: freeze vision; leave text as-is (or freeze it too if you want).

# Freeze vision encoder (as in your code)
for p in clip_model.visual.parameters():
    p.requires_grad = False

# If you want to freeze text encoder too, uncomment:
# for p in clip_model.transformer.parameters(): p.requires_grad = False
# for p in clip_model.token_embedding.parameters(): p.requires_grad = False
# for p in clip_model.ln_final.parameters(): p.requires_grad = False
# for name, p in clip_model.named_parameters():
#     if "text_projection" in name:
#         p.requires_grad = False

# CLIP normalization stats (OpenAI CLIP standard)
# preprocess() outputs normalized tensors with these constants.
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)

# Bounds in normalized space corresponding to pixel range [0,1]
CLIP_NORM_MIN = (0.0 - CLIP_MEAN) / CLIP_STD
CLIP_NORM_MAX = (1.0 - CLIP_MEAN) / CLIP_STD

# ----------------------------
# 3. Dataset
# ----------------------------
class CLIPPostDataset(Dataset):
    def __init__(self, csv_file, image_base, max_images=MAX_IMAGES):
        self.df = pd.read_csv(csv_file)
        self.image_base = image_base
        self.preprocess = preprocess
        self.max_images = max_images

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, fname: str) -> str:
        if not fname:
            return ""
        fname = str(fname)
        return fname if os.path.isabs(fname) else os.path.join(self.image_base, fname)

    def _open_image(self, path: str):
        try:
            if path and os.path.exists(path):
                return Image.open(path).convert("RGB")
        except Exception:
            pass
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["post"])
        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        image_list = []
        try:
            img_json = json.loads(row["pictures"])
            if isinstance(img_json, dict):
                image_list = [v for v in img_json.values() if v]
            elif isinstance(img_json, list):
                image_list = [x for x in img_json if x]
            elif isinstance(img_json, str):
                image_list = [img_json] if img_json else []
        except Exception:
            pass

        img_tensors = []
        for img_file in image_list[:self.max_images]:
            img = self._open_image(self._resolve_path(img_file))
            if img is not None:
                img_tensors.append(self.preprocess(img))

        dummy = self.preprocess(Image.new("RGB", (224, 224), (128, 128, 128)))
        while len(img_tensors) < self.max_images:
            img_tensors.append(dummy)

        imgs = torch.stack(img_tensors[:self.max_images])  # [N, 3, 224, 224] (CLIP-normalized)
        return text, imgs, label

# ----------------------------
# 4. Model Wrapper (supports adv text embeddings)
# ----------------------------
class CLIPBinaryClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.fc = nn.Linear(
            clip_model.visual.output_dim + clip_model.text_projection.shape[1], 1
        )
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def encode_text_from_embeddings(self, text_embeds, text_tokens):
        x = text_embeds + self.clip.positional_embedding.to(text_embeds.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).to(torch.float32)

        eot_idx = text_tokens.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_idx] @ self.clip.text_projection
        return text_features.to(torch.float32)

    def forward(self, images_batch, text_tokens, adv_text_embed=None):
        B, N, C, H, W = images_batch.shape
        images_batch = images_batch.view(B * N, C, H, W)

        image_features = self.clip.encode_image(images_batch).to(torch.float32)
        image_features = image_features.view(B, N, -1).mean(1)

        if adv_text_embed is None:
            text_features = self.clip.encode_text(text_tokens).to(torch.float32)
        else:
            text_features = self.encode_text_from_embeddings(adv_text_embed, text_tokens)

        combined = torch.cat([image_features, text_features], dim=1)
        logits = self.fc(combined)
        return logits.squeeze()

# ----------------------------
# 5. Prepare Data & Weights
# ----------------------------
dataset_full = CLIPPostDataset(csv_path, image_base_path)

total_rows = len(dataset_full)
sample_size = min(total_rows, 1048575)
subset_idx = random.sample(range(total_rows), sample_size)
dataset = torch.utils.data.Subset(dataset_full, subset_idx)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

print("Calculating class weights from training set...")
train_indices_mapped = [dataset.indices[i] for i in train_ds.indices]
train_labels = dataset_full.df.iloc[train_indices_mapped]["label"].values

num_neg = int((train_labels == 0).sum())
num_pos = int((train_labels == 1).sum())
print(f"  Training set stats: Negatives={num_neg}, Positives={num_pos}")

pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
pos_weight_tensor = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
print(f"  Using pos_weight: {pos_weight_val:.2f}")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ----------------------------
# 6. Joint PGD: perturb BOTH images (normalized space) and text embeddings
# ----------------------------
def clamp_clip_normalized(x_norm):
    return torch.max(torch.min(x_norm, CLIP_NORM_MAX), CLIP_NORM_MIN)

def gen_adv_joint_img_text(model, images_norm, text_tokens, labels, criterion,
                           img_eps, img_alpha, img_steps,
                           txt_eps, txt_alpha, txt_steps):
    """
    images_norm: CLIP-normalized images [B, N, 3, H, W]
    img_eps/img_alpha: pixel-space budgets; converted to normalized-space per-channel.
    text attack: delta in embedding space, clamped to [-txt_eps, txt_eps]
    """
    model.eval()

    # Convert pixel-space eps/alpha to normalized-space per channel
    img_eps_norm   = (img_eps / CLIP_STD)
    img_alpha_norm = (img_alpha / CLIP_STD)

    # --- image variable (normalized space) ---
    adv_images = images_norm.clone().detach().requires_grad_(True)

    # --- base text embeddings (no grad needed) ---
    with torch.no_grad():
        base_text_embed = model.clip.token_embedding(text_tokens).detach()  # [B, T, D]

    # mask PAD tokens (CLIP pad token id is 0)
    pad_mask = (text_tokens != 0).float().unsqueeze(-1)  # [B, T, 1]

    delta_txt = torch.zeros_like(base_text_embed).requires_grad_(True)

    joint_steps = max(img_steps, txt_steps)

    for step in range(joint_steps):
        adv_text_embed = base_text_embed + delta_txt * pad_mask

        logits = model(adv_images, text_tokens, adv_text_embed=adv_text_embed)
        loss = criterion(logits, labels)

        grad_img, grad_txt = torch.autograd.grad(
            loss, [adv_images, delta_txt], retain_graph=False, create_graph=False
        )

        # Image PGD update
        if step < img_steps:
            adv_images = adv_images + img_alpha_norm * torch.sign(grad_img)
            adv_images = torch.min(torch.max(adv_images, images_norm - img_eps_norm), images_norm + img_eps_norm)
            adv_images = clamp_clip_normalized(adv_images).detach().requires_grad_(True)

        # Text embedding PGD update
        if step < txt_steps:
            delta_txt = delta_txt + txt_alpha * torch.sign(grad_txt)
            delta_txt = torch.clamp(delta_txt, -txt_eps, txt_eps).detach().requires_grad_(True)

    model.train()
    adv_text_final = (base_text_embed + delta_txt * pad_mask).detach()
    return adv_images.detach(), adv_text_final

# ----------------------------
# 7. Training Setup (Clean + Adv)
# ----------------------------
model = CLIPBinaryClassifier(clip_model).to(device).float()

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable tensors: {len(trainable_params)}")
print("Trainable names:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print("  ", n)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.AdamW(trainable_params, lr=lr)

with open(log_filename, "w") as f:
    f.write("Epoch,Train_Loss,Val_F1,Val_BalAcc,Val_Prec,Val_Rec\n")

print("\nStarting Adversarial Training (Clean + Joint Adv Img+Txt)...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Clean+Adv Joint]")
    for texts, imgs, labels in progress_bar:
        texts_tok = clip.tokenize(texts, truncate=True).to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)   # clean (normalized)
        labels = labels.to(device, non_blocking=True)

        # 1) Generate joint adversarial pair (img + text)
        adv_imgs, adv_text_embed = gen_adv_joint_img_text(
            model, imgs, texts_tok, labels, criterion,
            img_eps=PGD_EPS, img_alpha=PGD_ALPHA, img_steps=PGD_STEPS,
            txt_eps=TXT_EPS, txt_alpha=TXT_ALPHA, txt_steps=TXT_STEPS
        )

        optimizer.zero_grad(set_to_none=True)

        # 2a) Clean loss (clean image + clean text)
        logits_clean = model(imgs, texts_tok, adv_text_embed=None)
        loss_clean = criterion(logits_clean, labels)

        # 2b) Adv loss (adv image + adv text)
        logits_adv = model(adv_imgs, texts_tok, adv_text_embed=adv_text_embed)
        loss_adv = criterion(logits_adv, labels)

        # Mixed loss
        loss = CLEAN_WEIGHT * loss_clean + ADV_WEIGHT * loss_adv

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{total_loss / (progress_bar.n + 1):.4f}"})

    avg_train_loss = total_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

    # --- Validation (Clean) ---
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for texts, imgs, labels in val_loader:
            texts_tok = clip.tokenize(texts, truncate=True).to(device)
            imgs = imgs.to(device)
            logits = model(imgs, texts_tok)
            preds = (torch.sigmoid(logits) > 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    f1 = f1_score(val_targets, val_preds, zero_division=0)
    bal_acc = balanced_accuracy_score(val_targets, val_preds)
    prec = precision_score(val_targets, val_preds, zero_division=0)
    rec = recall_score(val_targets, val_preds, zero_division=0)

    print(f"  Val F1: {f1:.4f} | Bal Acc: {bal_acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    with open(log_filename, "a") as f:
        f.write(f"{epoch+1},{avg_train_loss:.4f},{f1:.4f},{bal_acc:.4f},{prec:.4f},{rec:.4f}\n")

# ----------------------------
# 8. Evaluation (Clean + Joint Adv + ROC)
# ----------------------------
def evaluate_robustness_joint(model, val_loader, device,
                              img_eps, img_alpha, steps,
                              txt_eps, txt_alpha):
    print("\n--- Final Robustness Evaluation (Joint PGD Img+Txt) ---")
    model.eval()
    criterion_eval = nn.BCEWithLogitsLoss(reduction="mean")

    all_clean_preds, all_adv_preds = [], []
    all_clean_probs, all_adv_probs = [], []
    all_labels = []

    img_eps_norm   = (img_eps / CLIP_STD)
    img_alpha_norm = (img_alpha / CLIP_STD)

    for texts, imgs, labels in tqdm(val_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        text_tokens = clip.tokenize(texts, truncate=True).to(device)

        # Clean
        with torch.no_grad():
            clean_logits = model(imgs, text_tokens)
            clean_probs = torch.sigmoid(clean_logits)
            clean_preds = (clean_probs > 0.5).float()

        all_clean_preds.extend(clean_preds.cpu().numpy())
        all_clean_probs.extend(clean_probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Joint adversarial (recompute base embeddings)
        with torch.no_grad():
            base_text_embed = model.clip.token_embedding(text_tokens).detach()

        pad_mask = (text_tokens != 0).float().unsqueeze(-1)

        imgs_adv = imgs.clone().detach().requires_grad_(True)
        delta_txt = torch.zeros_like(base_text_embed).requires_grad_(True)

        for _ in range(steps):
            adv_text_embed = base_text_embed + delta_txt * pad_mask
            logits = model(imgs_adv, text_tokens, adv_text_embed=adv_text_embed)
            loss = criterion_eval(logits, labels)

            grad_img, grad_txt = torch.autograd.grad(loss, [imgs_adv, delta_txt])

            imgs_adv = imgs_adv + img_alpha_norm * torch.sign(grad_img)
            imgs_adv = torch.min(torch.max(imgs_adv, imgs - img_eps_norm), imgs + img_eps_norm)
            imgs_adv = clamp_clip_normalized(imgs_adv).detach().requires_grad_(True)

            delta_txt = delta_txt + txt_alpha * torch.sign(grad_txt)
            delta_txt = torch.clamp(delta_txt, -txt_eps, txt_eps).detach().requires_grad_(True)

        with torch.no_grad():
            adv_text_embed = base_text_embed + delta_txt * pad_mask
            adv_logits = model(imgs_adv, text_tokens, adv_text_embed=adv_text_embed)
            adv_probs = torch.sigmoid(adv_logits)
            adv_preds = (adv_probs > 0.5).float()

        all_adv_preds.extend(adv_preds.cpu().numpy())
        all_adv_probs.extend(adv_probs.cpu().numpy())

    clean_bal_acc = balanced_accuracy_score(all_labels, all_clean_preds)
    adv_bal_acc = balanced_accuracy_score(all_labels, all_adv_preds)
    clean_f1 = f1_score(all_labels, all_clean_preds, zero_division=0)
    adv_f1 = f1_score(all_labels, all_adv_preds, zero_division=0)

    return (
        clean_bal_acc, adv_bal_acc,
        clean_f1, adv_f1,
        np.array(all_clean_probs), np.array(all_adv_probs), np.array(all_labels)
    )

def plot_clean_vs_adv_roc(clean_probs, adv_probs, labels, save_path="roc_clean_vs_adv_joint.png"):
    if len(np.unique(labels)) < 2:
        print("⚠️ ROC undefined: Only one class present in test set.")
        return

    fpr_clean, tpr_clean, _ = roc_curve(labels, clean_probs)
    auc_clean = auc(fpr_clean, tpr_clean)

    fpr_adv, tpr_adv, _ = roc_curve(labels, adv_probs)
    auc_adv = auc(fpr_adv, tpr_adv)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_clean, tpr_clean, lw=2, label=f"Clean (AUC={auc_clean:.3f})")
    plt.plot(fpr_adv, tpr_adv, lw=2, label=f"Joint Adv (AUC={auc_adv:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Adversarially Trained (Clean+Adv; Joint Img+Txt PGD)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    print(f"✅ ROC figure saved to: {save_path}")

c_bal, a_bal, c_f1, a_f1, c_probs, a_probs, all_lbls = evaluate_robustness_joint(
    model, val_loader, device,
    img_eps=PGD_EPS, img_alpha=PGD_ALPHA, steps=EVAL_STEPS,
    txt_eps=TXT_EPS, txt_alpha=TXT_ALPHA
)

print(f"\nFinal Clean Balanced Acc: {c_bal:.4f}")
print(f"Final Joint Adv Balanced Acc: {a_bal:.4f}")
print(f"Final Clean F1: {c_f1:.4f}")
print(f"Final Joint Adv F1: {a_f1:.4f}")

plot_clean_vs_adv_roc(c_probs, a_probs, all_lbls)

torch.save(model.state_dict(), "clip_adv_train_clean_plus_adv_joint_img_text.pt")
print("✅ Training complete and model saved.")
