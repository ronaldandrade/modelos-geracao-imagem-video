"""Comparação didática em 2D: difusão (DDPM + amostrador DDIM) vs Flow Matching
(Rectified Flow) vs Rectified Flow com uma rodada de reflow.

Alvo: anel com 8 gaussianas (raio 2, desvio 0.08).

Métricas:
- retidão: comprimento do caminho / distância entre início e fim, medida numa
  trajetória fina (200 passos). Uma reta perfeita vale 1.0.
- precisão: fração de amostras a menos de 0.3 (cerca de 3.75 desvios) de algum centro.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim

NUM_CENTERS = 8
RADIUS = 2.0
NOISE_STD = 0.08
CENTERS = torch.stack(
    [
        RADIUS * torch.cos(torch.linspace(0, 2 * math.pi, NUM_CENTERS + 1)[:-1]),
        RADIUS * torch.sin(torch.linspace(0, 2 * math.pi, NUM_CENTERS + 1)[:-1]),
    ],
    dim=-1,
)


class ToyMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1))


def sample_target_ring(batch_size=512):
    idx = torch.randint(0, NUM_CENTERS, (batch_size,))
    return CENTERS[idx] + torch.randn(batch_size, 2) * NOISE_STD


def _train(loss_fn, steps, lr, seed):
    torch.manual_seed(seed)
    model = ToyMLP()
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = loss_fn(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model, loss.item()


# ---------------------------------------------------------------------------
# Flow Matching / Rectified Flow: x_t = (1 - t) x0 + t x1, x1 ~ N(0, I), alvo v = x1 - x0
# ---------------------------------------------------------------------------
def train_rectified_flow(steps=4000, lr=2e-3, batch_size=512, pairs=None, seed=0):
    def loss_fn(model):
        if pairs is None:
            x0 = sample_target_ring(batch_size)
            x1 = torch.randn(batch_size, 2)
        else:  # reflow: pares (ruído, dado) acoplados pela ODE do modelo anterior
            idx = torch.randint(0, pairs[0].shape[0], (batch_size,))
            x1, x0 = pairs[0][idx], pairs[1][idx]
        t = torch.rand(batch_size, 1)
        xt = (1.0 - t) * x0 + t * x1
        return torch.mean((model(xt, t) - (x1 - x0)) ** 2)

    return _train(loss_fn, steps, lr, seed)


@torch.no_grad()
def sample_rectified_flow(model, x1, num_steps):
    x, dt = x1.clone(), 1.0 / num_steps
    traj = [x.clone()]
    for i in range(num_steps):
        t = torch.full((x.shape[0], 1), 1.0 - i * dt)
        x = x - model(x, t) * dt  # Euler de t=1 (ruído) até t=0 (dado)
        traj.append(x.clone())
    return x, traj


# ---------------------------------------------------------------------------
# DDPM (predição de epsilon, schedule linear, T = 1000) com amostrador DDIM (eta = 0)
# ---------------------------------------------------------------------------
T_DDPM = 1000
BETAS = torch.linspace(1e-4, 0.02, T_DDPM)
ALPHA_BAR = torch.cumprod(1.0 - BETAS, dim=0)


def train_ddpm(steps=4000, lr=2e-3, batch_size=512, seed=0):
    def loss_fn(model):
        x0 = sample_target_ring(batch_size)
        t = torch.randint(0, T_DDPM, (batch_size,))
        eps = torch.randn(batch_size, 2)
        ab = ALPHA_BAR[t].unsqueeze(-1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return torch.mean((model(xt, t.float() / T_DDPM) - eps) ** 2)

    return _train(loss_fn, steps, lr, seed)


@torch.no_grad()
def sample_ddim(model, xT, num_steps):
    ts = torch.linspace(T_DDPM - 1, 0, num_steps + 1).round().long()
    x = xT.clone()
    traj = [x.clone()]
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        ab, ab_next = ALPHA_BAR[t], (ALPHA_BAR[t_next] if i < num_steps - 1 else torch.tensor(1.0))
        eps = model(x, torch.full((x.shape[0],), t.item() / T_DDPM))
        x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x = ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps
        traj.append(x.clone())
    return x, traj


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def straightness(traj):
    pts = torch.stack(traj)
    length = (pts[1:] - pts[:-1]).norm(dim=-1).sum(0)
    chord = (pts[-1] - pts[0]).norm(dim=-1)
    return (length / (chord + 1e-8)).mean().item()


def precision(samples, tol=0.3):
    d = torch.cdist(samples, CENTERS).min(dim=-1).values
    return (d < tol).float().mean().item()


def run_experiment():
    n_eval = 2000
    torch.manual_seed(123)
    noise = torch.randn(n_eval, 2)

    print("Treinando DDPM (predicao de ruido)...")
    ddpm, l_ddpm = train_ddpm()
    print(f"  loss final: {l_ddpm:.4f}")
    print("Treinando Rectified Flow (Flow Matching)...")
    rf, l_rf = train_rectified_flow()
    print(f"  loss final: {l_rf:.4f}  (perda alta e esperada: o alvo x1 - x0 tem variancia irredutivel)")
    print("Gerando pares acoplados pela ODE e treinando 1 rodada de reflow...")
    z = torch.randn(20000, 2)
    x_coupled, _ = sample_rectified_flow(rf, z, 200)
    reflow, l_re = train_rectified_flow(pairs=(z, x_coupled), seed=1)
    print(f"  loss final: {l_re:.4f}")

    samplers = {
        "DDPM + DDIM": lambda n: sample_ddim(ddpm, noise, n),
        "Rectified Flow": lambda n: sample_rectified_flow(rf, noise, n),
        "Rectified Flow + reflow": lambda n: sample_rectified_flow(reflow, noise, n),
    }

    print("\nRetidao da trajetoria (comprimento / corda, 200 passos; reta = 1.000):")
    for name, fn in samplers.items():
        _, traj = fn(200)
        print(f"  {name:<24s} {straightness(traj):.3f}")

    steps_list = [1, 2, 5, 10, 20, 50]
    print("\nPrecisao (fracao de amostras a < 0.3 de um centro) por numero de passos:")
    print("  " + " " * 24 + "".join(f"{s:>8d}" for s in steps_list))
    for name, fn in samplers.items():
        row = "".join(f"{precision(fn(s)[0]):8.3f}" for s in steps_list)
        print(f"  {name:<24s}{row}")

    print("\nLeitura: so os caminhos CONDICIONAIS do Flow Matching sao retos. O campo")
    print("marginal aprendido e curvo quando o alvo tem varios modos; o reflow")
    print("re-acopla ruido e dado e deixa a trajetoria mais reta, o que permite menos passos.")


if __name__ == "__main__":
    run_experiment()
