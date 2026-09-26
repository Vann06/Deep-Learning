"""Experimento con mas semillas: LSTM vs Transformer para H=24 y H=48.

Entrena ambos modelos con varias semillas (misma arquitectura, hiperparametros
y prediccion residual que Lab7_Task2/Lab7_Task3) y guarda las metricas de
validacion en results/semillas.json. Lo consume Lab7_Task4.ipynb.

Las clases son una copia compacta de las de los notebooks: se necesita un
modulo importable para entrenar en paralelo con multiprocessing en Windows.

Uso (desde app/):
    python experimento_semillas.py                # semillas 0 1 2 3 4 42
    python experimento_semillas.py 7 8 9          # semillas propias
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "ETTh1.csv"
OUT_PATH = HERE / "results" / "semillas.json"

L = 96
EPOCHS, BATCH_SIZE, LR = 20, 64, 1e-3
THREADS_PER_JOB = 2      # fijo: el numero de hilos cambia el orden de las sumas en punto flotante
N_WORKERS = 6
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 42]


def load_splits():
    # Mismo pipeline que el Task 1: z-score sobre OT completa y split 60/20/20 contiguo
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1)
    OT = data[:, -1].astype(np.float32)
    OT_norm = (OT - OT.mean()) / OT.std()
    n = len(OT_norm)
    n_train, n_val = int(0.60 * n), int(0.20 * n)
    OT_train = torch.from_numpy(OT_norm[:n_train].copy())
    OT_val = torch.from_numpy(OT_norm[n_train:n_train + n_val].copy())
    return OT_train, OT_val


def make_windows(series, L, H):
    # X^(t) = (x_{t-L+1}, ..., x_t),  y^(t) = x_{t+H}
    n_examples = series.numel() - L - H + 1
    X = series.unfold(0, L, 1)[:n_examples].clone()
    y = series[L + H - 1:L + H - 1 + n_examples].clone()
    return X, y


def make_pe(max_len, d_model):
    # PE(t, 2i) = sin(t / 10000^{2i/d}),  PE(t, 2i+1) = cos(t / 10000^{2i/d})
    t = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.pow(10000.0, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(t / div)
    pe[:, 1::2] = torch.cos(t / div)
    return pe


class LSTMForecaster(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.d_h = d_h
        k = 1.0 / math.sqrt(d_h)
        self.Wx = nn.Parameter(torch.empty(4 * d_h, d_in).uniform_(-k, k))
        self.Wh = nn.Parameter(torch.empty(4 * d_h, d_h).uniform_(-k, k))
        self.b = nn.Parameter(torch.empty(4 * d_h).uniform_(-k, k))
        self.W_out = nn.Parameter(torch.empty(1, d_h).uniform_(-k, k))
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, L = x.shape
        x = x.unsqueeze(-1)
        h = x.new_zeros(B, self.d_h)
        c = x.new_zeros(B, self.d_h)
        for t in range(L):
            gates = x[:, t, :] @ self.Wx.T + h @ self.Wh.T + self.b
            i, f, g, o = gates.chunk(4, dim=-1)
            i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
            c = f * c + i * g                  # c_t = f_t * c_{t-1} + i_t * g_t
            h = o * torch.tanh(c)              # h_t = o_t * tanh(c_t)
        return (h @ self.W_out.T + self.b_out).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, L):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.input_proj = nn.Linear(1, d_model)
        self.register_buffer("PE", make_pe(L, d_model))

        def xavier(*shape):
            return nn.Parameter(nn.init.xavier_uniform_(torch.empty(*shape)))
        self.WQ = xavier(d_model, d_model)
        self.WK = xavier(d_model, d_model)
        self.WV = xavier(d_model, d_model)
        self.WO = xavier(d_model, d_model)
        self.W1 = xavier(d_model, d_ff)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = xavier(d_ff, d_model)
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.gamma1 = nn.Parameter(torch.ones(d_model))
        self.beta1 = nn.Parameter(torch.zeros(d_model))
        self.gamma2 = nn.Parameter(torch.ones(d_model))
        self.beta2 = nn.Parameter(torch.zeros(d_model))
        self.W_out = nn.Linear(d_model, 1)

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return gamma * (x - mean) / torch.sqrt(var + eps) + beta

    def multi_head_attention(self, x):
        B, L, _ = x.shape
        Q = (x @ self.WQ).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = (x @ self.WK).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = (x @ self.WV).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        attn_w = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        out = (attn_w @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return out @ self.WO, attn_w

    def feed_forward(self, x):
        return F.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def forward(self, x):
        B, L = x.shape
        x = self.input_proj(x.unsqueeze(-1)) + self.PE[:L]
        att_out, _ = self.multi_head_attention(x)
        x = self.layer_norm(x + att_out, self.gamma1, self.beta1)
        x = self.layer_norm(x + self.feed_forward(x), self.gamma2, self.beta2)
        return self.W_out(x[:, 0, :]).squeeze(-1)


def run_job(job):
    model_name, H, seed = job
    # Determinismo: hilos fijos + algoritmos deterministas -> mismos numeros en cada corrida
    torch.set_num_threads(THREADS_PER_JOB)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()

    OT_train, OT_val = load_splits()
    X_tr, y_tr = make_windows(OT_train, L, H)
    X_vl, y_vl = make_windows(OT_val, L, H)
    naive_mae = torch.abs(y_vl - X_vl[:, -1]).mean().item()

    # Prediccion residual (igual que RESIDUAL=True en los notebooks):
    # X~ = X - x_t,  y~ = y - x_t,  y_hat = x_t + f(X~)
    X_in = X_tr - X_tr[:, -1:]
    y_in = y_tr - X_tr[:, -1]

    torch.manual_seed(seed)
    if model_name == "LSTM":
        model = LSTMForecaster(d_in=1, d_h=32)
    else:
        model = TransformerForecaster(d_model=32, n_heads=2, d_ff=64, L=L)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(X_in.shape[0])
        for start in range(0, X_in.shape[0], BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X_in[idx]), y_in[idx]).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_vl_in = X_vl - X_vl[:, -1:]
        pred = torch.cat([model(X_vl_in[i:i + 512]) for i in range(0, X_vl.shape[0], 512)])
        pred = pred + X_vl[:, -1]

    mae = torch.abs(y_vl - pred).mean().item()
    rmse = torch.sqrt(((y_vl - pred) ** 2).mean()).item()
    result = {"model": model_name, "H": H, "seed": seed, "mae": mae, "rmse": rmse,
              "ratio": mae / naive_mae, "segundos": round(time.time() - t0)}
    print(json.dumps(result), flush=True)
    return result


def main(seeds):
    # LSTM primero: es el mas lento, asi se reparte mejor la carga entre procesos
    jobs = [(m, H, s) for m in ("LSTM", "Transformer") for s in seeds for H in (24, 48)]
    with Pool(N_WORKERS) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump({"seeds": seeds, "threads_per_job": THREADS_PER_JOB, "results": results}, fh, indent=2)
    print("Resultados guardados en", OUT_PATH)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or DEFAULT_SEEDS)
