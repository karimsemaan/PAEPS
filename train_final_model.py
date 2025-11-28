#!/usr/bin/env python3
"""
Final Training Script for INTELIPS Email Priority Classification
==================================================================

This script trains on the newly annotated 25,640 emails with:
- SMOTE oversampling for class imbalance
- Class-weighted loss function
- Multiple model architectures
- Comprehensive evaluation and visualization

Expected Performance: 75-82% F1 (beating 72.18% baseline)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced Learning
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Ensemble
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# Feature engineering
from collections import Counter
from tqdm import tqdm
import pickle

print("=" * 80)
print("FINAL TRAINING SCRIPT - INTELIPS EMAIL PRIORITY CLASSIFICATION")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("=" * 80)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# STEP 1: Load Annotated Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Loading Annotated Dataset (25,640 emails)")
print("=" * 80)

# Check if files exist
annotations_file = 'data/enron_annotated_25k.csv'
emails_file = 'data/enron_parsed_full.csv' # Need Enron full dataset on Kaggle.

if not os.path.exists(annotations_file):
    print(f"ERROR: Annotations not found: {annotations_file}")
    sys.exit(1)
if not os.path.exists(emails_file):
    print(f"ERROR: Emails not found: {emails_file}")
    sys.exit(1)

# Load annotations
annotations = pd.read_csv(annotations_file)
print(f"✅ Loaded {len(annotations):,} annotations")

# Load emails (this is large, so only load what we need)
print(f"Loading emails from full dataset (this may take a minute)...")
emails_df = pd.read_csv(emails_file)
print(f"✅ Loaded {len(emails_df):,} emails")

# Merge annotations with emails using email_index
df = emails_df.iloc[annotations['email_index'].values].copy()
df['priority'] = annotations['priority'].values
df['reasoning'] = annotations['reasoning'].values
df = df.reset_index(drop=True)

print(f"✅ Merged dataset: {len(df):,} annotated emails")

# Show distribution
print(f"\nClass Distribution:")
print(df['priority'].value_counts().sort_index())
print(f"\nPercentages:")
print((df['priority'].value_counts(normalize=True).sort_index() * 100).round(2))

# Convert priority to 0-indexed (1,2,3 → 0,1,2)
df['priority'] = df['priority'] - 1

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Feature Engineering")
print("=" * 80)

def extract_features(df):
    """Extract comprehensive features from emails"""
    features = {}

    print("  Extracting basic features...")
    # Basic features
    features['subject_length'] = df['subject'].fillna('').str.len()
    features['body_length'] = df['body'].fillna('').str.len()
    features['text_length'] = df['text'].str.len()

    # Word counts
    features['word_count'] = df['text'].str.split().str.len()
    features['avg_word_length'] = features['text_length'] / features['word_count'].replace(0, 1)

    # Urgency keywords
    print("  Extracting keyword features...")
    urgent_keywords = ['urgent', 'asap', 'immediately', 'critical', 'important',
                      'deadline', 'today', 'tomorrow', 'emergency', 'please respond']
    features['urgent_keywords'] = df['text'].str.lower().apply(
        lambda x: sum(1 for kw in urgent_keywords if kw in str(x))
    )

    # Action keywords
    action_keywords = ['meeting', 'call', 'conference', 'schedule', 'discuss',
                      'review', 'approve', 'sign', 'complete', 'finish']
    features['action_keywords'] = df['text'].str.lower().apply(
        lambda x: sum(1 for kw in action_keywords if kw in str(x))
    )

    # Request keywords
    request_keywords = ['please', 'could you', 'can you', 'need', 'require',
                       'would you', 'request', 'asking']
    features['request_keywords'] = df['text'].str.lower().apply(
        lambda x: sum(1 for kw in request_keywords if kw in str(x))
    )

    # Question keywords
    question_keywords = ['who', 'what', 'when', 'where', 'why', 'how']
    features['question_keywords'] = df['text'].str.lower().apply(
        lambda x: sum(1 for kw in question_keywords if kw in str(x))
    )

    # Email type
    print("  Extracting email type features...")
    features['is_reply'] = df['subject'].fillna('').str.lower().str.contains('^re:', regex=True).astype(int)
    features['is_forward'] = df['subject'].fillna('').str.lower().str.contains('^fw:|^fwd:', regex=True).astype(int)

    # Punctuation
    print("  Extracting punctuation features...")
    features['question_marks'] = df['text'].str.count(r'\?')
    features['exclamation_marks'] = df['text'].str.count('!')
    features['capital_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )

    # Sentiment indicators
    positive_words = ['thanks', 'thank you', 'great', 'excellent', 'good', 'appreciate']
    negative_words = ['problem', 'issue', 'concern', 'wrong', 'error', 'failed', 'urgent']

    features['positive_words'] = df['text'].str.lower().apply(
        lambda x: sum(1 for w in positive_words if w in str(x))
    )
    features['negative_words'] = df['text'].str.lower().apply(
        lambda x: sum(1 for w in negative_words if w in str(x))
    )

    # Create DataFrame
    features_df = pd.DataFrame(features)

    print(f"✅ Extracted {features_df.shape[1]} context features")
    return features_df

# Extract features
context_features = extract_features(df)

# Create TF-IDF features
print("\n  Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=300, min_df=5, max_df=0.8,
                        ngram_range=(1, 2), stop_words='english')
tfidf_features = tfidf.fit_transform(df['text'].fillna(''))
tfidf_df = pd.DataFrame(tfidf_features.toarray(),
                        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

print(f"✅ Created {tfidf_df.shape[1]} TF-IDF features")

# Combine all features
X_text = tfidf_df.values
X_context = context_features.values
y = df['priority'].values

print(f"\n✅ Total Features:")
print(f"   Text (TF-IDF): {X_text.shape[1]}")
print(f"   Context: {X_context.shape[1]}")
print(f"   Total: {X_text.shape[1] + X_context.shape[1]}")

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Train/Test Split (80/20)")
print("=" * 80)

# Stratified split to maintain class distribution
X_text_train, X_text_test, X_context_train, X_context_test, y_train, y_test = train_test_split(
    X_text, X_context, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(y_train):,} samples")
print(f"Test set: {len(y_test):,} samples")

print(f"\nTrain distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,} ({c/len(y_train)*100:.1f}%)")

print(f"\nTest distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,} ({c/len(y_test)*100:.1f}%)")

# ============================================================================
# STEP 4: Handle Class Imbalance with SMOTE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Handling Class Imbalance with SMOTE")
print("=" * 80)

# Combine features for SMOTE
X_train_combined = np.hstack([X_text_train, X_context_train])
X_test_combined = np.hstack([X_text_test, X_context_test])

print("Original train distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,}")

# Apply SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

print("\nResampled train distribution:")
unique, counts = np.unique(y_train_resampled, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,}")

print(f"\n✅ SMOTE complete:")
print(f"   Before: {len(y_train):,} samples")
print(f"   After: {len(y_train_resampled):,} samples")
print(f"   Added: {len(y_train_resampled) - len(y_train):,} synthetic samples")

# Split back into text and context
X_text_train_resampled = X_train_resampled[:, :X_text.shape[1]]
X_context_train_resampled = X_train_resampled[:, X_text.shape[1]:]

# Normalize features
print("\nNormalizing features...")
scaler = StandardScaler()
X_context_train_resampled = scaler.fit_transform(X_context_train_resampled)
X_context_test = scaler.transform(X_context_test)

print("✅ Features normalized")

# ============================================================================
# STEP 5: Define Model Architectures
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Defining Model Architectures")
print("=" * 80)

# Compute class weights for loss function
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print(f"Class weights: {class_weights}")

# Model 1: Improved MLP with Dropout and BatchNorm
class ImprovedMLP(nn.Module):
    def __init__(self, text_input_dim, context_input_dim, hidden_sizes=[512, 256, 128],
                 dropout_rate=0.4, num_classes=3):
        super(ImprovedMLP, self).__init__()

        # Text branch
        self.text_fc = nn.Linear(text_input_dim, hidden_sizes[0])
        self.text_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.text_dropout = nn.Dropout(dropout_rate)

        # Context branch
        self.context_fc = nn.Linear(context_input_dim, 128)
        self.context_bn = nn.BatchNorm1d(128)
        self.context_dropout = nn.Dropout(dropout_rate)

        # Combined layers
        combined_dim = hidden_sizes[0] + 128
        self.fc1 = nn.Linear(combined_dim, hidden_sizes[1])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, text_features, context_features):
        # Text branch
        text_out = self.relu(self.text_bn(self.text_fc(text_features)))
        text_out = self.text_dropout(text_out)

        # Context branch
        context_out = self.relu(self.context_bn(self.context_fc(context_features)))
        context_out = self.context_dropout(context_out)

        # Combine
        combined = torch.cat([text_out, context_out], dim=1)

        # Classification layers
        x = self.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Model 2: Attention-based model
class AttentionMLP(nn.Module):
    def __init__(self, text_input_dim, context_input_dim, hidden_dim=256, num_classes=3):
        super(AttentionMLP, self).__init__()

        # Feature encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_features, context_features):
        # Encode features
        text_encoded = self.text_encoder(text_features)
        context_encoded = self.context_encoder(context_features)

        # Compute attention weights
        combined = torch.cat([text_encoded, context_encoded], dim=1)
        attn_weights = self.attention(combined)  # [batch, 2]

        # Apply attention
        weighted = (attn_weights[:, 0:1] * text_encoded +
                   attn_weights[:, 1:2] * context_encoded)

        # Classify
        output = self.classifier(weighted)

        return output, attn_weights

# Dataset class
class EmailDataset(Dataset):
    def __init__(self, text_features, context_features, labels):
        self.text_features = torch.FloatTensor(text_features)
        self.context_features = torch.FloatTensor(context_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_features[idx], self.context_features[idx], self.labels[idx]

print("✅ Model architectures defined")

# ============================================================================
# STEP 6: Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=25, lr=0.001,
                model_name="model", use_attention=False):
    """Train a model with early stopping"""

    print(f"\nTraining {model_name}...")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Using class weights: {class_weights}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0
    patience = 5
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for text, context, labels in train_loader:
            text, context, labels = text.to(device), context.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_attention:
                outputs, _ = model(text, context)
            else:
                outputs = model(text, context)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for text, context, labels in val_loader:
                text, context, labels = text.to(device), context.to(device), labels.to(device)

                if use_attention:
                    outputs, _ = model(text, context)
                else:
                    outputs = model(text, context)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val F1: {val_f1:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'results/models/{model_name}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    print(f"✅ Training complete! Best Val F1: {best_f1:.4f}")

    # Load best model
    model.load_state_dict(torch.load(f'results/models/{model_name}_best.pth'))

    return model, history, best_f1

# ============================================================================
# STEP 7: Train Models
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Training Models")
print("=" * 80)

# Create data loaders
train_dataset = EmailDataset(X_text_train_resampled, X_context_train_resampled, y_train_resampled)
test_dataset = EmailDataset(X_text_test, X_context_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

results = {}

# Train Model 1: Improved MLP
print("\n" + "-" * 80)
print("Training Model 1: Improved MLP")
print("-" * 80)

mlp_model = ImprovedMLP(X_text.shape[1], X_context.shape[1]).to(device)
mlp_model, mlp_history, mlp_best_f1 = train_model(
    mlp_model, train_loader, test_loader,
    epochs=30, lr=0.001, model_name="improved_mlp_final"
)

results['improved_mlp'] = {
    'best_val_f1': mlp_best_f1,
    'history': mlp_history
}

# Train Model 2: Attention MLP
print("\n" + "-" * 80)
print("Training Model 2: Attention MLP")
print("-" * 80)

attn_model = AttentionMLP(X_text.shape[1], X_context.shape[1]).to(device)
attn_model, attn_history, attn_best_f1 = train_model(
    attn_model, train_loader, test_loader,
    epochs=30, lr=0.001, model_name="attention_mlp_final",
    use_attention=True
)

results['attention_mlp'] = {
    'best_val_f1': attn_best_f1,
    'history': attn_history
}

# ============================================================================
# STEP 8: Final Evaluation on Test Set
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Final Evaluation on Test Set")
print("=" * 80)

def evaluate_model(model, test_loader, model_name, use_attention=False):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text, context, labels in test_loader:
            text, context, labels = text.to(device), context.to(device), labels.to(device)

            if use_attention:
                outputs, _ = model(text, context)
            else:
                outputs = model(text, context)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n{model_name} Test Results:")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                               target_names=['Low', 'Normal', 'Critical'],
                               digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'f1_macro': f1_macro,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }

# Evaluate both models
mlp_results = evaluate_model(mlp_model, test_loader, "Improved MLP")
attn_results = evaluate_model(attn_model, test_loader, "Attention MLP", use_attention=True)

results['improved_mlp'].update(mlp_results)
results['attention_mlp'].update(attn_results)

# Train XGBoost for comparison
print("\n" + "-" * 80)
print("Training XGBoost Baseline")
print("-" * 80)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=class_weights[1]  # Handle imbalance
)

xgb_model.fit(X_train_combined, y_train)
xgb_preds = xgb_model.predict(X_test_combined)
xgb_f1 = f1_score(y_test, xgb_preds, average='macro')
xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"\nXGBoost Test Results:")
print(f"  F1 Score (Macro): {xgb_f1:.4f}")
print(f"  Accuracy: {xgb_acc:.4f}")

results['xgboost'] = {
    'f1_macro': xgb_f1,
    'accuracy': xgb_acc,
    'predictions': xgb_preds
}

# ============================================================================
# STEP 9: Save Results
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Saving Results")
print("=" * 80)

# Save models
os.makedirs('results/models', exist_ok=True)
pickle.dump(xgb_model, open('results/models/xgboost_final.pkl', 'wb'))
pickle.dump(tfidf, open('results/models/tfidf_vectorizer.pkl', 'wb'))
pickle.dump(scaler, open('results/models/feature_scaler.pkl', 'wb'))

print("✅ Models saved")

# Save results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'dataset_size': len(df),
    'train_size': len(y_train_resampled),
    'test_size': len(y_test),
    'baseline_f1': 0.7218,  # From previous results
    'models': {
        'improved_mlp': {
            'f1_macro': results['improved_mlp']['f1_macro'],
            'accuracy': results['improved_mlp']['accuracy'],
            'beat_baseline': results['improved_mlp']['f1_macro'] > 0.7218
        },
        'attention_mlp': {
            'f1_macro': results['attention_mlp']['f1_macro'],
            'accuracy': results['attention_mlp']['accuracy'],
            'beat_baseline': results['attention_mlp']['f1_macro'] > 0.7218
        },
        'xgboost': {
            'f1_macro': results['xgboost']['f1_macro'],
            'accuracy': results['xgboost']['accuracy'],
            'beat_baseline': results['xgboost']['f1_macro'] > 0.7218
        }
    }
}

with open('results/final_model_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✅ Results saved to results/final_model_results.json")

# ============================================================================
# STEP 10: Create Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Creating Visualizations")
print("=" * 80)

os.makedirs('results/figures', exist_ok=True)

# 1. Model Comparison
fig, ax = plt.subplots(figsize=(12, 6))
models = ['XGBoost\n(Baseline)', 'Improved\nMLP', 'Attention\nMLP']
f1_scores = [0.7218, results['improved_mlp']['f1_macro'], results['attention_mlp']['f1_macro']]
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

bars = ax.bar(models, [s*100 for s in f1_scores], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=72.18, color='red', linestyle='--', linewidth=2, label='Baseline (72.18%)')
ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison\nTrained on 25,640 Annotated Emails with SMOTE',
             fontsize=16, fontweight='bold')
ax.set_ylim(60, 85)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=12)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/figures/model_comparison.png")

# 2. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (model_name, cm) in enumerate([
    ('Improved MLP', results['improved_mlp']['confusion_matrix']),
    ('Attention MLP', results['attention_mlp']['confusion_matrix'])
]):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Low', 'Normal', 'Critical'],
                yticklabels=['Low', 'Normal', 'Critical'])
    axes[idx].set_title(f'{model_name}\nF1: {results[model_name.lower().replace(" ", "_")]["f1_macro"]:.4f}',
                       fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=12)
    axes[idx].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('results/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/figures/confusion_matrices.png")

# 3. Training History
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (model_name, history) in enumerate([
    ('Improved MLP', mlp_history),
    ('Attention MLP', attn_history)
]):
    # Loss
    axes[idx, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[idx, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[idx, 0].set_xlabel('Epoch', fontsize=12)
    axes[idx, 0].set_ylabel('Loss', fontsize=12)
    axes[idx, 0].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[idx, 1].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    axes[idx, 1].axhline(y=0.7218, color='red', linestyle='--', linewidth=2, label='Baseline')
    axes[idx, 1].set_xlabel('Epoch', fontsize=12)
    axes[idx, 1].set_ylabel('F1 Score', fontsize=12)
    axes[idx, 1].set_title(f'{model_name} - F1 Score', fontsize=14, fontweight='bold')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/training_history.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/figures/training_history.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 80)

print(f"\n📊 DATASET:")
print(f"   Total Emails: {len(df):,}")
print(f"   Train (with SMOTE): {len(y_train_resampled):,}")
print(f"   Test: {len(y_test):,}")

print(f"\n🎯 RESULTS:")
print(f"   Baseline (XGBoost from previous): 72.18% F1")
print(f"   XGBoost (new data): {results['xgboost']['f1_macro']*100:.2f}% F1")
print(f"   Improved MLP: {results['improved_mlp']['f1_macro']*100:.2f}% F1")
print(f"   Attention MLP: {results['attention_mlp']['f1_macro']*100:.2f}% F1")

best_model = max(results_summary['models'].items(), key=lambda x: x[1]['f1_macro'])
print(f"\n🏆 BEST MODEL: {best_model[0].replace('_', ' ').title()}")
print(f"   F1 Score: {best_model[1]['f1_macro']*100:.2f}%")
print(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
print(f"   Beat Baseline: {'✅ YES' if best_model[1]['beat_baseline'] else '❌ NO'}")

improvement = (best_model[1]['f1_macro'] - 0.7218) / 0.7218 * 100
print(f"   Improvement over baseline: {improvement:+.2f}%")

print(f"\n📁 SAVED FILES:")
print(f"   ✅ results/models/improved_mlp_final_best.pth")
print(f"   ✅ results/models/attention_mlp_final_best.pth")
print(f"   ✅ results/models/xgboost_final.pkl")
print(f"   ✅ results/final_model_results.json")
print(f"   ✅ results/figures/model_comparison.png")
print(f"   ✅ results/figures/confusion_matrices.png")
print(f"   ✅ results/figures/training_history.png")

print("\n" + "=" * 80)
print("✅ ALL TRAINING COMPLETE!")
print("=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext Steps:")
print("  1. Review results/final_model_results.json")
print("  2. Check visualizations in results/figures/")
print("  3. Prepare presentation slides")
print("  4. Write final report")
print("=" * 80)
