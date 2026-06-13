import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc
)

# 1. Custom KDE Naive Bayes Classifier
class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=0.2, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        total_samples = len(y)
        
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / total_samples
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X_c)
            self.models_[c] = kde
        return self
        
    def predict_proba(self, X):
        logprobs = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            log_density = self.models_[c].score_samples(X)
            logprobs[:, i] = log_density + np.log(self.priors_[c])
            
        # Log-sum-exp trick to avoid underflow/overflow
        probs = np.exp(logprobs - np.max(logprobs, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

# 2. Robustness Curves Plotting Functions
def plot_macro_roc(models, X_test, y_test, num_classes, output_path):
    plt.figure(figsize=(10, 8))
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
            else:
                y_score = model.decision_function(X_test)
                y_score = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
                
            fpr = dict()
            tpr = dict()
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= num_classes
            
            macro_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f'{name} (AUC = {macro_auc:.3f})', lw=2)
        except Exception as e:
            print(f"Could not plot ROC for {name}: {e}")
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - Macro Average OvR')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_lift_gain_curves(models, X_test, y_test, target_class, class_name, results_dir):
    y_test_bin = (y_test == target_class).astype(int)
    total_positives = np.sum(y_test_bin)
    
    if total_positives == 0:
        print(f"No positive samples for target class {target_class} ({class_name})")
        return

    # 1. Gain plot
    plt.figure(figsize=(10, 8))
    plt.plot([0, 100], [0, 100], 'k--', label='Random Model')

    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, target_class]
            else:
                y_score = model.decision_function(X_test)
                y_score = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
                probs = y_score[:, target_class]

            sorted_indices = np.argsort(probs)[::-1]
            sorted_y_test = y_test_bin[sorted_indices]
            
            cum_positives = np.cumsum(sorted_y_test)
            cum_gain = (cum_positives / total_positives) * 100
            sample_percentages = (np.arange(1, len(sorted_y_test) + 1) / len(sorted_y_test)) * 100
            
            plt.plot(sample_percentages, cum_gain, label=f'{name}', lw=2)
        except Exception as e:
            print(f"Could not plot Gain for {name}: {e}")
            
    plt.xlabel('% of Sample')
    plt.ylabel('% of Gain (Positives Captured)')
    plt.title(f'Cumulative Gain Chart - Class: {class_name}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/gain_curves.png', dpi=300)
    plt.close()

    # 2. Lift plot
    plt.figure(figsize=(10, 8))
    plt.axhline(y=1.0, color='k', linestyle='--', label='Random Model')
    overall_positive_rate = total_positives / len(y_test_bin)

    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, target_class]
            else:
                y_score = model.decision_function(X_test)
                y_score = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
                probs = y_score[:, target_class]

            sorted_indices = np.argsort(probs)[::-1]
            sorted_y_test = y_test_bin[sorted_indices]
            
            cum_positives = np.cumsum(sorted_y_test)
            cum_positive_rates = cum_positives / np.arange(1, len(sorted_y_test) + 1)
            cum_lift = cum_positive_rates / overall_positive_rate
            sample_percentages = (np.arange(1, len(sorted_y_test) + 1) / len(sorted_y_test)) * 100
            
            plt.plot(sample_percentages, cum_lift, label=f'{name}', lw=2)
        except Exception as e:
            print(f"Could not plot Lift for {name}: {e}")
            
    plt.xlabel('% of Sample')
    plt.ylabel('Lift')
    plt.title(f'Cumulative Lift Chart - Class: {class_name}')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/lift_curves.png', dpi=300)
    plt.close()

def main():
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    print("1. Loading dataset...")
    df = pd.read_csv('fer2013.csv')
    y_raw = df['emotion'].values
    pixels_raw = df['pixels'].values
    
    print("Parsing pixels into feature matrix...")
    n_samples = len(pixels_raw)
    n_features = 48 * 48
    X_raw = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for idx, p_str in enumerate(pixels_raw):
        X_raw[idx] = np.fromstring(p_str, dtype=np.float32, sep=' ')
    
    # Scale pixel features to [0, 1] for fast convergence of SVM and Logistic Regression
    X_raw /= 255.0
    
    # Emotion labels mapping based on standard FER2013 classes
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    class_names = [emotion_dict[i] for i in sorted(emotion_dict.keys())]
    
    print("Splitting dataset into train, test, and application subsets...")
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(X_raw, y_raw, train_size=3500, random_state=42, stratify=y_raw)
    X_test, X_app, y_test, y_app = train_test_split(X_temp, y_temp, train_size=1000, test_size=500, random_state=42, stratify=y_temp)
    
    print(f"Data shapes - Train: {X_train_full.shape}, Test: {X_test.shape}, App: {X_app.shape}")
    
    print("\n2. Applying Predictor Selection (Feature Selection)...")
    # VarianceThreshold scaled for normalized features (threshold=0.01)
    vt = VarianceThreshold(threshold=0.01)
    X_train_vt = vt.fit_transform(X_train_full)
    X_test_vt = vt.transform(X_test)
    X_app_vt = vt.transform(X_app)
    
    K = 150
    selector = SelectKBest(score_func=f_classif, k=K)
    X_train_sel = selector.fit_transform(X_train_vt, y_train_full)
    X_test_sel = selector.transform(X_test_vt)
    X_app_sel = selector.transform(X_app_vt)
    
    vt_support = vt.get_support()
    sel_support = selector.get_support()
    
    selected_original_indices = []
    vt_feature_idx = 0
    for idx, kept in enumerate(vt_support):
        if kept:
            if sel_support[vt_feature_idx]:
                selected_original_indices.append(idx)
            vt_feature_idx += 1
            
    scores = selector.scores_
    pvalues = selector.pvalues_
    
    feature_meta = []
    vt_idx = 0
    for original_idx, kept in enumerate(vt_support):
        if kept:
            score = scores[vt_idx]
            pval = pvalues[vt_idx]
            selected = sel_support[vt_idx]
            feature_meta.append({
                "Pixel Index": original_idx,
                "ANOVA Score": score,
                "p-value": pval,
                "Status": "Kept" if selected else "Eliminated"
            })
            vt_idx += 1
        else:
            feature_meta.append({
                "Pixel Index": original_idx,
                "ANOVA Score": 0.0,
                "p-value": 1.0,
                "Status": "Eliminated (Low Var)"
            })
            
    df_features = pd.DataFrame(feature_meta)
    df_features['Rank'] = df_features['ANOVA Score'].rank(ascending=False, method='min')
    df_features.sort_values(by='Rank', inplace=True)
    df_features.to_csv(f'{results_dir}/feature_selection_details.csv', index=False)
    
    top_15_features_table = df_features.head(15)
    print("\nTop 15 Predictors Ranked by ANOVA F-Score:")
    print(top_15_features_table.to_string(index=False))
    
    feature_names = [f"Pixel_{i}" for i in selected_original_indices]
    
    print("\n3. Initializing and training 11 classifiers with scaled features...")
    # Initialize classifiers with tuned hyperparameters for better performance and speed
    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "KDE Naive Bayes": KDEClassifier(bandwidth=0.2), # Smaller bandwidth for normalized features
        "Linear Discriminant": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=4, random_state=42),
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8), n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=42),
        "Linear SVM": SVC(kernel='linear', C=1.0, probability=False, random_state=42),
        "Gaussian SVM": SVC(kernel='rbf', C=5.0, gamma='scale', probability=False, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    }
    
    accuracies = {}
    precisions = {}
    recalls = {}
    f1_scores = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_sel, y_train_full)
        y_pred = model.predict(X_test_sel)
        predictions[name] = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        accuracies[name] = acc
        precisions[name] = prec
        recalls[name] = rec
        f1_scores[name] = f1
        print(f"-> Accuracy: {acc:.4f} | F1 (Macro): {f1:.4f}")
        
    df_metrics = pd.DataFrame({
        "Model": list(models.keys()),
        "Accuracy": [accuracies[m] for m in models],
        "Precision (Macro)": [precisions[m] for m in models],
        "Recall (Macro)": [recalls[m] for m in models],
        "F1-Score (Macro)": [f1_scores[m] for m in models]
    })
    df_metrics.sort_values(by='Accuracy', ascending=False, inplace=True)
    df_metrics.to_csv(f'{results_dir}/model_metrics_comparison.csv', index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Accuracy', y='Model', data=df_metrics, palette='viridis')
    plt.title('Classifier Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.xlim(0, 0.65)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/accuracy_comparison.png', dpi=300)
    plt.close()
    
    print("\n4. Generating confusion matrices...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    fig.delaxes(axes[11])
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrices.png', dpi=300)
    plt.close()
    
    print("\n5. Plotting Decision Tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(models["Decision Tree"], max_depth=3, filled=True, 
              feature_names=feature_names, class_names=class_names, fontsize=8)
    plt.title("Decision Tree Visualization (Max Depth Truncated to 3)")
    plt.tight_layout()
    plt.savefig(f'{results_dir}/decision_tree.png', dpi=300)
    plt.close()
    
    print("\n6. Plotting Feature Importances...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Decision Tree importances
    dt_importances = models["Decision Tree"].feature_importances_
    dt_indices = np.argsort(dt_importances)[::-1][:15]
    sns.barplot(x=dt_importances[dt_indices], y=[feature_names[i] for i in dt_indices], ax=ax1, palette='mako')
    ax1.set_title("Top 15 Features (Decision Tree)")
    ax1.set_xlabel("Importance Score")
    
    # Random Forest importances
    rf_importances = models["Random Forest"].feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1][:15]
    sns.barplot(x=rf_importances[rf_indices], y=[feature_names[i] for i in rf_indices], ax=ax2, palette='flare')
    ax2.set_title("Top 15 Features (Random Forest)")
    ax2.set_xlabel("Importance Score")
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/feature_importances.png', dpi=300)
    plt.close()
    
    print("\n7. Calculating Classification Error Tables...")
    error_records = []
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        for i, class_label in enumerate(class_names):
            total_samples = np.sum(cm[i, :])
            correct_samples = cm[i, i]
            misclassified = total_samples - correct_samples
            error_rate = (misclassified / total_samples) * 100 if total_samples > 0 else 0
            error_records.append({
                "Model": name,
                "Class": class_label,
                "Total Samples": total_samples,
                "Misclassified": misclassified,
                "Error Rate (%)": error_rate
            })
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(f'{results_dir}/classification_errors_by_class.csv', index=False)
    
    print("\n8. Generating Robustness Curves (ROC, Lift, Gain)...")
    plot_macro_roc(models, X_test_sel, y_test, len(class_names), f'{results_dir}/roc_curves.png')
    plot_lift_gain_curves(models, X_test_sel, y_test, 3, "Happy", results_dir)
    
    print("\n9. Generating 2D PCA Error Plot...")
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_sel)
    
    best_model_name = df_metrics.iloc[0]["Model"]
    print(f"Best model selected for PCA error visualization: {best_model_name}")
    y_pred_best = predictions[best_model_name]
    correct_mask = (y_test == y_pred_best)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test_pca[correct_mask, 0], X_test_pca[correct_mask, 1], c='green', label='Correctly Classified', alpha=0.6, s=25)
    plt.scatter(X_test_pca[~correct_mask, 0], X_test_pca[~correct_mask, 1], c='red', label='Misclassified', alpha=0.6, s=25)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA 2D Projection of Classifications ({best_model_name})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/pca_errors.png', dpi=300)
    plt.close()
    
    print("\n10. Running Prediction on the Application Set...")
    best_model = models[best_model_name]
    y_app_pred = best_model.predict(X_app_sel)
    
    df_app_results = pd.DataFrame({
        "Sample Index": np.arange(1, len(y_app) + 1),
        "True Emotion ID": y_app,
        "True Emotion Name": [emotion_dict[val] for val in y_app],
        "Predicted Emotion ID": y_app_pred,
        "Predicted Emotion Name": [emotion_dict[val] for val in y_app_pred],
        "Success": (y_app == y_app_pred)
    })
    df_app_results.to_csv(f'{results_dir}/predictions_aplicare.csv', index=False)
    
    app_accuracy = accuracy_score(y_app, y_app_pred)
    print(f"Application Set Accuracy: {app_accuracy:.4f}")
    
    print("\nAll pipeline tasks executed successfully! Results saved to 'results/' directory.")

if __name__ == '__main__':
    main()
