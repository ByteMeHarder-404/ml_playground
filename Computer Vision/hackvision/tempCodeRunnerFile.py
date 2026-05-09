import torch
from datasets import load_dataset, concatenate_datasets
from PIL import Image

# ==========================================
# 1. GLOBAL MASTER SCHEMA & MAPPINGS
# ==========================================
ANATOMY_LABELS = {
    "lung": 0, "bone": 1, "dental": 2, "brain": 3, "eye": 4, 
    "skin": 5, "breast": 6, "pathology": 7, "abdomen": 8, "blood": 9
}

MODALITY_LABELS = {
    "xray": 0, "ct": 1, "mri": 2, "fundus": 3, "dermoscopy": 4, 
    "mammography": 5, "microscopy": 6
}

# The Master Disease Vector (60 Slots)
# Mapping: [0-13]: Lung | [14]: Bone | [15-21]: Skin | [22-26]: Eye | [27-30]: Brain 
# [31-34]: Dental | [35-36]: Breast | [37-45]: Pathology | [46-53]: Blood | [54-64]: Abdomen
MAX_DISEASE_SLOTS = 65 

NIH_CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
               "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
               "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

DENTEX_CLASSES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]

# ==========================================
# 2. ROBUST TYPE-AWARE TRANSFORMATION
# ==========================================

def universal_transform(example, anatomy, modality, offset, num_classes, dataset_id, labels_map=None):
    """Handles string/int labels and standardizes metadata."""
    disease_vec = [0.0] * MAX_DISEASE_SLOTS
    
    # Extract labels from common keys
    raw_val = example.get('label', example.get('labels', example.get('diagnosis', 0)))
    work_list = raw_val if isinstance(raw_val, list) else [raw_val]
    label_text_list = []

    for val in work_list:
        if val is None: continue
        idx = -1
        
        # Case A: Label is a string (NIH style)
        if isinstance(val, str) and labels_map:
            if val in labels_map:
                idx = labels_map.index(val)
                label_text_list.append(val)
        
        # Case B: Label is an integer (Standard style)
        elif isinstance(val, (int, float)):
            idx = int(val)
            if labels_map and idx < len(labels_map):
                label_text_list.append(labels_map[idx])
            else:
                label_text_list.append(str(idx))

        # Populate Vector
        if 0 <= idx < num_classes:
            disease_vec[offset + idx] = 1.0

    return {
        "image": example['image'],
        "anatomy": ANATOMY_LABELS[anatomy],
        "modality": MODALITY_LABELS[modality],
        "disease_label": disease_vec,
        "metadata": {
            "dataset_id": dataset_id,
            "view": example.get('view_position', "standard"),
            "age": str(example.get('age', "unknown")),
            "sex": str(example.get('sex', "unknown")),
            "localization": example.get('localization', anatomy),
            "original_labels_text": ", ".join(label_text_list) if label_text_list else "Normal/No Finding"
        }
    }

# ==========================================
# 3. MASTER INGESTION PIPELINE (VERIFIED REPOS)
# ==========================================

def create_unified_medical_dataset():
    print("--- Starting Verified Multi-Expert Data Ingestion ---")

    expert_configs = {
        "lung":    {"path": "BahaaEldin0/NIH-Chest-Xray-14", "mod": "xray", "off": 0, "cls": 14, "map": NIH_CLASSES},
        "bone":    {"path": "KhalfounMehdi/MURA", "mod": "xray", "off": 14, "cls": 1, "map": ["Abnormal"]},
        "skin":    {"path": "marmal88/skin_cancer", "mod": "dermoscopy", "off": 15, "cls": 7, "map": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]},
        "eye":     {"path": "danjacobellis/retinamnist_224", "mod": "fundus", "off": 22, "cls": 5, "map": ["L0", "L1", "L2", "L3", "L4"]},
        "brain":   {"path": "Simezu/brain-tumour-MRI-scan", "mod": "mri", "off": 27, "cls": 4, "map": ["glioma", "meningioma", "no_tumor", "pituitary"]},
        "dental":  {"path": "ibrahimhamamci/DENTEX", "mod": "xray", "off": 31, "cls": 4, "map": DENTEX_CLASSES},
        "breast":  {"path": "Subramanian123/breastmnist", "mod": "mammography", "off": 35, "cls": 2, "map": ["Malignant", "Benign"]},
        "path":    {"path": "Subramanian123/pathmnist", "mod": "microscopy", "off": 37, "cls": 9, "map": ["adi", "bac", "deb", "lym", "muc", "mus", "norm", "str", "tum"]},
        "blood":   {"path": "Subramanian123/bloodmnist", "mod": "microscopy", "off": 46, "cls": 8, "map": ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]},
        "abdomen": {"path": "Subramanian123/organamnist", "mod": "ct", "off": 54, "cls": 11, "map": None}
    }

    segments = []
    for anatomy, cfg in expert_configs.items():
        try:
            print(f"-> Processing: {anatomy.upper()}...")
            ds = load_dataset(cfg['path'], split="train")
            
            std_ds = ds.map(
                lambda x: universal_transform(x, anatomy, cfg['mod'], cfg['off'], cfg['cls'], cfg['path'], cfg['map']),
                remove_columns=ds.column_names,
                desc=f"Standardizing {anatomy}"
            )
            segments.append(std_ds)
            print(f"   Done: {len(std_ds)} images.")
        except Exception as e:
            print(f"!! Failed {anatomy}: {e}")

    master_ds = concatenate_datasets(segments).shuffle(seed=42)
    print(f"\n--- FINAL MASTER DATASET: {len(master_ds)} IMAGES ---")
    return master_ds

if __name__ == "__main__":
    dataset = create_unified_medical_dataset()
    
    # Verification Peek
    sample = dataset[0]
    print(f"\n[VERIFICATION]")
    print(f"Anatomy Type: {sample['anatomy']} (Dental=2, Brain=3, Skin=5)")
    print(f"Label Found: {sample['metadata']['original_labels_text']}")
    print(f"Disease Vector (Sample): {sample['disease_label'][:40]}")