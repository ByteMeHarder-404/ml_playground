# Global Medical Navigator & Specialist System

This repository implements a **Hybrid Interpretable AI System** for medical diagnosis, designed to solve the challenge of multi-modal, multi-anatomy disease classification. It moves beyond simple classification by integrating **Visual Navigation**, **Memory-Based Retrieval (Visual RAG)**, and **Mixture-of-Experts (MoE)** routing.

## System Architecture

The system is composed of four distinct, loosely coupled modules that mimic a clinical workflow: **Perception (Navigator)** -> **Recall (Memory)** -> **Triage (Gating)** -> **Diagnosis (Specialists)**.

### Architecture Diagram

graph TD
    subgraph "1. Perception & Navigation"
        Img["Input Image"] --> Nav["Global Medical Navigator<br/>DenseNet121"]
        Nav -->|Features| Emb["Visual Embedding z"]
        Nav -->|Logits| AnaP["Anatomy Probs"]
        Nav -->|Logits| ModP["Modality Probs"]
    end

    subgraph "2. Visual Memory (RAG)"
        Emb --> FAISS["FAISS Index<br/>Visual Memory"]
        AnaP -->|Filter| FAISS
        FAISS -->|Retrieve| Neighbors["Top-K Neighbors"]
        Neighbors -->|Weighted Avg| KNN["KNN Disease Prior"]
    end

    subgraph "3. Intelligent Routing"
        AnaP --> Gate["Medical Gating Network"]
        ModP --> Gate
        KNN --> Gate
        Gate -->|Gating Weights| Router{"Router"}
        AnaP -->|Identity Boost| Router
    end

    subgraph "4. Specialist Execution"
        Router -->|Lung Case| Exp1["Lung Expert<br/>DenseNet121"]
        Router -->|Skin Case| Exp2["Skin Expert<br/>MobileNetV2"]
        Router -->|Eye Case| Exp3["Eye Expert<br/>EfficientNetB0"]
        Router -->|Other| Exp4["General Expert<br/>ResNet50"]
    end

    Exp1 --> Final["Final Disease Prediction"]
    Exp2 --> Final
    Exp3 --> Final
    Exp4 --> Final

    style Nav fill:#e1f5fe,stroke:#01579b
    style FAISS fill:#fff3e0,stroke:#e65100
    style Gate fill:#f3e5f5,stroke:#4a148c
    style Router fill:#f3e5f5,stroke:#4a148c
    style Exp1 fill:#e8f5e9,stroke:#2e7d32
    style Exp2 fill:#e8f5e9,stroke:#2e7d    32
    style Exp3 fill:#e8f5e9,stroke:#2e7d32
    style Exp4 fill:#e8f5e9,stroke:#2e7d32

---

## Deep Dive: Implementation & Design Choices

### 1. Unified Data Ingestion ("The Universal Transform")
One of the biggest challenges was unifying disparate datasets (NIH X-ray, MURA Bone, HAM10000 Skin, etc.) that have different label formats.
- **Solution**: We implemented a `universal_transform` function that maps all inputs to a **Master Disease Vector (65 slots)**.
- **Schema**:
    - Slots 0-13: Lung (NIH classes)
    - Slot 14: Bone (Abnormal)
    - Slots 15-21: Skin (Melanoma, Nevus, etc.)
    - ...and so on for Eye, Brain, Dental, Breast, Pathology, Blood, and Abdomen.
- **Impact**: This allows the system to train on a massive, diverse dataset without architectural changes for each new source.

### 2. Global Medical Navigator
- **Backbone Choice**: `DenseNet121`. We chose DenseNet over ResNet for the navigator because of its feature reuse capability, which is crucial for capturing fine-grained anatomical details across different modalities (X-ray vs MRI).
- **Multi-Head Output**:
    - **Embedding Head**: Projects features to a 512-dim latent space (`z`). Includes `LayerNorm` to stabilize the embedding distribution for vector search.
    - **Anatomy & Modality Heads**: Separate linear layers to explicitly classify *what* part of the body and *which* imaging type is present.

### 3. Improved Retrieval Engine (Visual RAG)
We use **FAISS** for high-speed similarity search, but with specific modifications for medical accuracy:
- **Oversample & Filter Strategy**: Standard KNN is blind to semantic constraints. We query `K*5` neighbors and then strictly filter them using the Navigator's predicted anatomy. This prevents a "Chest X-ray" query from retrieving a visually similar "Dental X-ray".
- **Inverse-Distance Weighted Prior**: The "KNN Prior" passed to the gating network is not a simple average. It is weighted by `1 / (distance + epsilon)`, giving far more influence to highly similar historical cases.

### 4. Medical Gating Network & "Identity Boost"
The Gating Network decides which specialist to use. It takes a concatenated input of **Anatomy Probs + Modality Probs + KNN Prior**.
- **The "Identity Boost" Fix**: Purely learned gating can be unstable early in training. We implemented a residual-like connection:
  ```python
  boosted_weights = 0.5 * gate_weights + 0.5 * anatomy_probs
  ```
  This forces the router to respect the Navigator's strong anatomy classification (e.g., if it's 99% Lung, send to Lung Expert) while allowing the Gate to learn subtle exceptions based on the KNN context.

### 5. High-Velocity Specialist Team
We employ a **Mixture-of-Experts (MoE)** approach where each expert is architecturally tailored to its domain:
- **Lung Expert (`DenseNet121`)**: Proven SOTA for Chest X-rays (CheXNet architecture).
- **Skin Expert (`MobileNetV2`)**: Lightweight and efficient, suitable for the distinct, high-contrast features of dermoscopy.
- **Eye Expert (`EfficientNetB0`)**: Excellent at capturing fine-grained retinal details with fewer parameters.
- **General Expert (`ResNet50`)**: A robust, general-purpose backbone for diverse tasks (Pathology, Abdomen, etc.).

## Explainability (LIME Integration)
The system integrates **LIME** to generate heatmaps.
- **Context-Aware Perturbation**: When generating explanations, we broadcast the **KNN Prior** across all perturbed samples. This ensures that the explanation reflects the model's behavior *given the specific retrieved context*, rather than explaining the model in isolation.
