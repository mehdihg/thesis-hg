from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# --- 1. بارگذاری مدل‌ها ---
structure_best = load_model('MultiView-RumorDetection/best_models7913.hdf5')
contex_best    = load_model('MultiView-RumorDetection/best_modeln8000.hdf5')
net_best       = load_model('MultiView-RumorDetection/best_modelsn84347.hdf5')

# --- 2. آماده‌سازی داده‌های آزمون ---
# subtreefeature[917:], data[917:], X[917:] و y_test از کد شما

subtreefeature = np.load("subtreefeature.npy")
data            = np.load("data.npy")
X               = np.load("X.npy")
y_test          = np.load("y_test.npy")

# حالا ببرید مثل قبل:
X_struct = subtreefeature[917:]
X_text   = data[917:]
X_net    = X[917:]
y_true   = np.argmax(y_test, axis=1)



X_struct = subtreefeature[917:]
X_text   = data[917:]
X_net    = X[917:]        # بعد از reshape: (num_samples, 1, feature_dim)
y_true   = np.argmax(y_test, axis=1)  # تبدیل one-hot به لیبل عددی

# --- 3. پیش‌بینی هر مدل ---
pred_s = np.argmax(structure_best.predict(X_struct), axis=1)
pred_c = np.argmax(contex_best   .predict(X_text ), axis=1)
pred_n = np.argmax(net_best      .predict(X_net  ), axis=1)

# --- 4. تابع کمکی برای چاپ گزارش ---
def report(name, y_true, y_pred):
    print(f"===== {name} =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='binary'))
    print("Recall   :", recall_score   (y_true, y_pred, average='binary'))
    print("F1-Score :", f1_score       (y_true, y_pred, average='binary'))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    print("\n")

# --- 5. چاپ گزارش برای هر مدل ---
report("Structure-Model (GRU)", y_true, pred_s)
report("Context-Model (CNN)",   y_true, pred_c)
report("Centrality-Model (LSTM)", y_true, pred_n)

# (اختیاری) اگر مدل ترکیبی را هم دارید:
ensemble_best = load_model('MultiView-RumorDetection/best_modelsn.hdf5')
pred_e = np.argmax(
    ensemble_best.predict([X_struct, X_net, X_text]), axis=1
)
report("Multi-View Ensemble", y_true, pred_e)
