# qml_embedding_benchmark.py

import time
import csv
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import pennylane as qml
import pennylane.numpy as pnp


# =========================
# 1) Cấu hình thí nghiệm
# =========================
SEED = 42
np.random.seed(SEED)
pnp.random.seed(SEED)

N_WIRES = 2
N_LAYERS = 2
EPOCHS = 25
LEARNING_RATE = 0.08

OUTPUT_CSV = "qml_embedding_results.csv"

# Toy dataset 2 lớp, 2 feature
X, y = make_moons(n_samples=240, noise=0.15, random_state=SEED)

# Scale về [0, 1]
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

y_train = y_train.astype(float)
y_test = y_test.astype(int)

# PennyLane simulator
dev = qml.device("default.qubit", wires=N_WIRES)


# =========================
# 2) Hàm đổi 2 feature -> state vector 4 chiều
# =========================
def to_state_vector(x):
    x0, x1 = float(x[0]), float(x[1])

    vec = np.array([x0, x1, x0 * x1, 1.0], dtype=float)
    norm = np.linalg.norm(vec)

    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    return vec / norm


# =========================
# 3) Tạo QNode
# =========================
@qml.qnode(dev, interface="autograd")
def circuit(x, weights, embed_method):
    if embed_method == "angle":
        qml.AngleEmbedding(np.pi * x, wires=range(N_WIRES), rotation="Y")

    elif embed_method == "amplitude":
        state = to_state_vector(x)
        qml.AmplitudeEmbedding(state, wires=range(N_WIRES), normalize=False)

    elif embed_method == "stateprep":
        state = to_state_vector(x)
        qml.StatePrep(state, wires=range(N_WIRES))

    else:
        raise ValueError(f"Unknown method: {embed_method}")

    qml.StronglyEntanglingLayers(weights, wires=range(N_WIRES))
    return qml.expval(qml.PauliZ(0))


# =========================
# 4) Loss và predict
# =========================
def expval_to_prob(expval):
    return 0.5 * (expval + 1.0)


def binary_cross_entropy(prob, target):
    prob = pnp.clip(prob, 1e-7, 1 - 1e-7)
    return -(target * pnp.log(prob) + (1.0 - target) * pnp.log(1.0 - prob))


def loss_fn(weights, X_batch, y_batch, embed_method):
    losses = []
    for x, y_true in zip(X_batch, y_batch):
        expval = circuit(x, weights, embed_method)
        prob = expval_to_prob(expval)
        losses.append(binary_cross_entropy(prob, y_true))
    return pnp.mean(pnp.stack(losses))


def predict_label(X_data, weights, embed_method):
    preds = []
    for x in X_data:
        expval = circuit(x, weights, embed_method)
        prob = float(expval_to_prob(expval))
        preds.append(1 if prob >= 0.5 else 0)
    return np.array(preds, dtype=int)


# =========================
# 5) Train/evaluate một method
# =========================
def run_experiment(embed_method):
    weights = pnp.array(
        np.random.normal(0.0, 0.1, size=(N_LAYERS, N_WIRES, 3)),
        requires_grad=True,
    )

    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        weights = opt.step(
            lambda w: loss_fn(w, X_train, y_train, embed_method),
            weights
        )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred_train = predict_label(X_train, weights, embed_method)
    y_pred_test = predict_label(X_test, weights, embed_method)
    infer_time = time.perf_counter() - t1

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    final_loss = float(loss_fn(weights, X_train, y_train, embed_method))

    return {
        "method": embed_method,
        "loss": final_loss,
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "train_time_s": float(train_time),
        "infer_time_s": float(infer_time),
    }


# =========================
# 6) Nhập method từ người dùng
# =========================
print("Chọn method để chạy:")
print("1 - angle")
print("2 - amplitude")
print("3 - stateprep")
print("all - chạy cả 3 method")

choice = input("Nhập lựa chọn của bạn: ").strip().lower()

if choice == "1":
    METHODS = ["angle"]
elif choice == "2":
    METHODS = ["amplitude"]
elif choice == "3":
    METHODS = ["stateprep"]
elif choice == "all":
    METHODS = ["angle", "amplitude", "stateprep"]
else:
    raise ValueError("Lựa chọn không hợp lệ. Hãy nhập 1, 2, 3 hoặc all.")


# =========================
# 7) Chạy benchmark và lưu CSV
# =========================
results = []

for method in METHODS:
    print(f"\n=== Running: {method} ===")
    res = run_experiment(method)
    results.append(res)
    print(
        f"{method:10s} | "
        f"loss={res['loss']:.4f} | "
        f"train_acc={res['train_acc']:.4f} | "
        f"test_acc={res['test_acc']:.4f}"
    )

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "loss", "train_acc", "test_acc"])
    writer.writeheader()
    for res in results:
        writer.writerow({
            "method": res["method"],
            "loss": res["loss"],
            "train_acc": res["train_acc"],
            "test_acc": res["test_acc"],
        })

print(f"\nĐã lưu kết quả vào: {OUTPUT_CSV}")

print("\n=== SUMMARY ===")
for res in results:
    print(
        f"{res['method']:10s} | "
        f"loss={res['loss']:.4f} | "
        f"train_acc={res['train_acc']:.4f} | "
        f"test_acc={res['test_acc']:.4f}"
    )