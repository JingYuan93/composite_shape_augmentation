#!/usr/bin/env python3
import os
import random
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    exponents = -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    freqs = torch.exp(exponents)
    args_ = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args_), torch.cos(args_)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

class ResBlock1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=1)
        self.c2 = nn.Conv1d(c, c, 3, padding=1)
        self.g1 = nn.GroupNorm(1, c)
        self.g2 = nn.GroupNorm(1, c)
    def forward(self, x):
        h = F.relu(self.g1(self.c1(x)))
        h = self.g2(self.c2(h))
        return F.relu(h + x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.LayerNorm(c)
        self.a = nn.MultiheadAttention(c, heads, batch_first=True)
    def forward(self, x):
        h = self.n(x)
        y, _ = self.a(h, h, h)
        return x + y

class DiffUNet(nn.Module):
    def __init__(self, in_dim, hid_dim, seq_len):
        super().__init__()
        self.tm = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.cm = nn.Sequential(nn.Linear(in_dim * seq_len, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.ip = nn.Conv1d(in_dim, hid_dim, 1)
        self.gi = nn.GroupNorm(1, hid_dim)
        self.rb = nn.ModuleList([ResBlock1D(hid_dim) for _ in range(4)])
        self.ab = nn.ModuleList([SelfAttentionBlock(hid_dim) for _ in range(2)])
        self.go = nn.GroupNorm(1, hid_dim)
        self.op = nn.Conv1d(hid_dim, in_dim, 1)
    def forward(self, x, t, cond=None):
        B, L, C = x.shape
        temb = get_timestep_embedding(t, args.hid_dim)
        temb = self.tm(temb)
        if cond is None:
            cemb = torch.zeros_like(temb)
        else:
            flat = cond.view(B, -1)
            cemb = self.cm(flat)
        emb = (temb + cemb).unsqueeze(-1)
        h = x.permute(0, 2, 1)
        h = self.gi(self.ip(h))
        h = h + emb
        for rb in self.rb:
            h = rb(h)
        h2 = h.permute(0, 2, 1)
        for ab in self.ab:
            h2 = ab(h2)
        h3 = h2.permute(0, 2, 1)
        h3 = self.go(h3)
        out = self.op(h3)
        return out.permute(0, 2, 1)

def save_generated(gen, case, model_name):
    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"{case}_{model_name}.csv"
    rows = []
    for seq in gen:
        arr = seq.detach().cpu().numpy() * 19.0 + 6.0
        rows.append(arr.reshape(-1, args.feat_dim))
    pd.DataFrame(np.vstack(rows)).to_csv(os.path.join(args.out_dir, fname), header=False, index=False)

def train_gan(data, case):
    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.noise_dim, args.hid_dim, 3, batch_first=True)
            self.o = nn.Linear(args.hid_dim, args.feat_dim)
        def forward(self, z):
            h, _ = self.l(z)
            return torch.sigmoid(self.o(h))
    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.feat_dim, args.hid_dim, 3, batch_first=True)
            self.o = nn.Linear(args.hid_dim, 1)
        def forward(self, x):
            h, _ = self.l(x)
            return torch.sigmoid(self.o(h[:, -1]))
    Gm = G().to(device)
    Dm = D().to(device)
    oG = torch.optim.Adam(Gm.parameters(), lr=5e-4)
    oD = torch.optim.Adam(Dm.parameters(), lr=5e-4)
    bce = nn.BCELoss()
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True)
    for _ in range(200):
        for real, in loader:
            bs = real.size(0)
            for __ in range(20):
                z = torch.randn(bs, args.seq_len, args.noise_dim, device=device)
                fake = Gm(z)
                adv = bce(Dm(fake), torch.ones(bs,1,device=device))
                dif = fake[:,1:,:] - fake[:,:-1,:]
                lm = torch.mean(F.relu(-dif)**2)
                dif2 = dif[:,1:,:] - dif[:,:-1,:]
                lc = torch.mean(F.relu(-dif2)**2)
                l = adv + args.lambda_mono*lm + args.lambda_curv*lc
                oG.zero_grad(); l.backward(); oG.step()
            with torch.no_grad():
                fake = Gm(torch.randn(bs, args.seq_len, args.noise_dim, device=device))
            dr = Dm(real); df = Dm(fake)
            ld = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
            oD.zero_grad(); ld.backward(); oD.step()
    Gm.eval()
    with torch.no_grad():
        z = torch.randn(args.num_samples, args.seq_len, args.noise_dim, device=device)
        gen = Gm(z).clamp(0,1)
        gen, _ = torch.cummax(gen, dim=1)
    save_generated(gen, case, "GAN")

def train_ae(data, case):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.LSTM(args.feat_dim, args.hid_dim, 3, batch_first=True)
            self.d = nn.LSTM(args.hid_dim, args.hid_dim, 3, batch_first=True)
            self.o = nn.Linear(args.hid_dim, args.feat_dim)
        def forward(self, x):
            _, (h, _) = self.e(x)
            rep = h[-1].unsqueeze(1).repeat(1, args.seq_len, 1)
            y, _ = self.d(rep)
            return torch.sigmoid(self.o(y))
    model = M().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)
    for _ in range(200):
        for x, in loader:
            xh = model(x)
            lrec = mse(xh, x)
            dif = xh[:,1:,:] - xh[:,:-1,:]
            lm = torch.mean(F.relu(-dif)**2)
            dif2 = dif[:,1:,:] - dif[:,:-1,:]
            lc = torch.mean(F.relu(-dif2)**2)
            loss = lrec + args.lambda_mono*lm + args.lambda_curv*lc
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    codes = []
    with torch.no_grad():
        for i in range(data.size(0)):
            _, (h, _) = model.e(data[i:i+1])
            codes.append(h[-1].squeeze(0))
    codes = torch.stack(codes)
    gens = []
    for _ in range(args.num_samples):
        i, j = np.random.choice(len(codes), 2, replace=False)
        a = np.random.rand()
        z = a*codes[i] + (1-a)*codes[j] + 0.02*torch.randn_like(codes[0])
        rep = z.unsqueeze(0).unsqueeze(1).repeat(1, args.seq_len, 1)
        y, _ = model.d(rep)
        y = torch.sigmoid(model.o(y))
        y, _ = torch.cummax(y, dim=1)
        gens.append(y.squeeze(0))
    gen = torch.stack(gens)
    save_generated(gen, case, "AE")

def train_vae(data, case):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.LSTM(args.feat_dim, args.hid_dim, 3, batch_first=True)
            self.f1 = nn.Linear(args.hid_dim, args.z_dim)
            self.f2 = nn.Linear(args.hid_dim, args.z_dim)
            self.di = nn.Linear(args.z_dim, args.hid_dim)
            self.d = nn.LSTM(args.hid_dim, args.hid_dim, 3, batch_first=True)
            self.o = nn.Linear(args.hid_dim, args.feat_dim)
        def encode(self, x):
            _, (h, _) = self.e(x)
            h = h[-1]
            return self.f1(h), self.f2(h)
        def reparam(self, mu, lv):
            std = torch.exp(0.5*lv)
            eps = torch.randn_like(std)
            return mu + eps*std
        def decode(self, z):
            h0 = self.di(z).unsqueeze(0)
            rep = h0.repeat(args.seq_len,1,1).permute(1,0,2)
            y, _ = self.d(rep)
            return torch.sigmoid(self.o(y))
        def forward(self, x):
            mu, lv = self.encode(x)
            z = self.reparam(mu, lv)
            return self.decode(z), mu, lv
    model = M().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)
    for _ in range(200):
        for x, in loader:
            xh, mu, lv = model(x)
            lrec = mse(xh, x)
            kld = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
            dif = xh[:,1:,:] - xh[:,:-1,:]
            lm = torch.mean(F.relu(-dif)**2)
            dif2 = dif[:,1:,:] - dif[:,:-1,:]
            lc = torch.mean(F.relu(-dif2)**2)
            loss = lrec + 0.05*kld + args.lambda_mono*lm + args.lambda_curv*lc
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    gens = []
    with torch.no_grad():
        for _ in range(args.num_samples):
            z = torch.randn(args.z_dim, device=device)
            y = model.decode(z)
            y, _ = torch.cummax(y, dim=1)
            gens.append(y)
    gen = torch.stack(gens)
    save_generated(gen, case, "VAE")

def train_timegan(data, case):
    class E(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.feat_dim, args.hid_dim, 3, batch_first=True)
        def forward(self, x):
            return self.l(x)[0]
    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.hid_dim, args.hid_dim, 3, batch_first=True)
            self.o = nn.Linear(args.hid_dim, args.feat_dim)
        def forward(self, h):
            return torch.sigmoid(self.o(self.l(h)[0]))
    class Gm(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.noise_dim, args.hid_dim, 3, batch_first=True)
        def forward(self, z):
            return self.l(z)[0]
    class S(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.hid_dim, args.hid_dim, 3, batch_first=True)
        def forward(self, h):
            return self.l(h)[0]
    class Dm(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.LSTM(args.hid_dim, args.hid_dim, 3, batch_first=True)
            self.o = nn.Linear(args.hid_dim, 1)
        def forward(self, h):
            return torch.sigmoid(self.o(self.l(h)[0][:,-1]))
    E1, R1, G1, S1, D1 = E().to(device), R().to(device), Gm().to(device), S().to(device), Dm().to(device)
    oE = torch.optim.Adam(list(E1.parameters())+list(R1.parameters()), lr=5e-4)
    oG = torch.optim.Adam(list(G1.parameters())+list(S1.parameters()), lr=5e-4)
    oD = torch.optim.Adam(D1.parameters(), lr=5e-4)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True)
    for _ in range(150):
        for x, in loader:
            l = mse(R1(E1(x)), x)
            oE.zero_grad(); l.backward(); oE.step()
    for _ in range(150):
        for x, in loader:
            h = E1(x)
            l = mse(S1(h)[:,1:,:], h[:,:-1,:])
            oG.zero_grad(); l.backward(); oG.step()
    for _ in range(200):
        for x, in loader:
            bs = x.size(0)
            for __ in range(20):
                z = torch.randn(bs, args.seq_len, args.noise_dim, device=device)
                hf = G1(z); hs = S1(hf); xf = R1(hs)
                gu = bce(D1(hs), torch.ones(bs,1,device=device))
                gs = mse(S1(hs)[:,1:,:], hs[:,:-1,:])
                gr = mse(xf, x)
                dif = xf[:,1:,:] - xf[:,:-1,:]; lm = torch.mean(F.relu(-dif)**2)
                dif2 = dif[:,1:,:] - dif[:,:-1,:]; lc = torch.mean(F.relu(-dif2)**2)
                loss = gu + 100*gs + 100*gr + args.lambda_mono*lm + args.lambda_curv*lc
                oG.zero_grad(); loss.backward(); oG.step()
            with torch.no_grad():
                hr = E1(x)
                hf = G1(torch.randn(bs, args.seq_len, args.noise_dim, device=device))
                hs = S1(hf)
            dr, df = D1(hr), D1(hs)
            ld = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
            oD.zero_grad(); ld.backward(); oD.step()
    with torch.no_grad():
        z = torch.randn(args.num_samples, args.seq_len, args.noise_dim, device=device)
        xf = R1(S1(G1(z))).clamp(0,1)
        xf, _ = torch.cummax(xf, dim=1)
    save_generated(xf, case, "TimeGAN")

def train_diffusion(data, case):
    deltas = data[:,1:,:] - data[:,:-1,:]
    mean_delta = deltas.mean(dim=0, keepdim=True)
    model = DiffUNet(args.feat_dim, args.hid_dim, args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.LR)
    loader = DataLoader(TensorDataset(data), batch_size=args.BATCH_SZ, shuffle=True)
    mse = nn.MSELoss()
    t_lin = torch.linspace(0,1,args.seq_len,device=device).view(1,args.seq_len,1)
    betas = torch.linspace(args.BETA_START, args.BETA_END, args.TIMESTEPS, device=device)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, dim=0)
    for _ in range(args.EPOCHS):
        for x0_norm, in loader:
            bs = x0_norm.size(0)
            t = torch.randint(0, args.TIMESTEPS, (bs,), device=device)
            a_t = abar[t].view(bs,1,1)
            noise = torch.randn_like(x0_norm)
            x_t = torch.sqrt(a_t)*x0_norm + torch.sqrt(1-a_t)*noise
            pred_noise = model(x_t, t)
            loss_n = mse(pred_noise, noise)
            x_pred0 = (x_t - torch.sqrt(1-a_t)*pred_noise)/torch.sqrt(a_t)
            loss_r = mse(x_pred0, x0_norm)
            dif = x_pred0[:,1:,:] - x_pred0[:,:-1,:]
            loss_m = torch.mean(F.relu(-dif)**2)
            dif2 = dif[:,1:,:] - dif[:,:-1,:]
            loss_c = torch.mean(F.relu(-dif2)**2)
            b0 = F.relu(-x_pred0); b1 = F.relu(x_pred0-1)
            loss_rg = torch.mean(b0**2 + b1**2)
            start = x0_norm[:,0:1,:]; end = x0_norm[:,-1:,:]
            baseline = (1 - t_lin)*start + t_lin*end
            loss_i = mse(x_pred0, baseline)
            loss = (args.lambda_noise*loss_n + args.lambda_recon*loss_r +
                    args.lambda_mono*loss_m + args.lambda_curv*loss_c +
                    args.lambda_range*loss_rg + args.lambda_interp*loss_i)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
    gens = []
    with torch.no_grad():
        for _ in range(args.num_samples):
            start = data[0:1,0:1,:]; end = data[0:1,-1:,:]
            baseline = ((1-t_lin)*start + t_lin*end).squeeze(0)
            aT = abar[-1]
            x = torch.sqrt(aT)*baseline + torch.sqrt(1-aT)*torch.randn_like(baseline)
            for ti in reversed(range(args.TIMESTEPS)):
                beta = betas[ti]; a = alphas[ti]; ab = abar[ti]
                eps = model(x.unsqueeze(0), torch.tensor([ti], device=device)).squeeze(0)
                x = (x - beta/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                x = x.clamp(0,1)
            x, _ = torch.cummax(x, dim=0)
            x[0,:] = 0; x[-1,:] = 1
            gens.append(x)
    gen_norm = torch.stack(gens, 0)
    save_generated(gen_norm, case, "Diffusion")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",         default="combined_matrix.csv", type=str)
    parser.add_argument("--num_raw_samples",   default=3,                     type=int)
    parser.add_argument("--seq_len",           default=10,                    type=int)
    parser.add_argument("--noise_dim",         default=64,                    type=int)
    parser.add_argument("--hid_dim",           default=128,                   type=int)
    parser.add_argument("--z_dim",             default=64,                    type=int)
    parser.add_argument("--TIMESTEPS",         default=1000,                  type=int)
    parser.add_argument("--BETA_START",        default=1e-4,                  type=float)
    parser.add_argument("--BETA_END",          default=0.02,                  type=float)
    parser.add_argument("--LR",                default=1e-5,                  type=float)
    parser.add_argument("--EPOCHS",            default=200,                   type=int)
    parser.add_argument("--BATCH_SZ",          default=8,                     type=int)
    parser.add_argument("--seed",              default=42,                    type=int)
    parser.add_argument("--lambda_noise",      default=1.0,                   type=float)
    parser.add_argument("--lambda_recon",      default=1.0,                   type=float)
    parser.add_argument("--lambda_mono",       default=0.1,                   type=float)
    parser.add_argument("--lambda_curv",       default=0.1,                   type=float)
    parser.add_argument("--lambda_range",      default=1.0,                   type=float)
    parser.add_argument("--lambda_interp",     default=10.0,                  type=float)
    parser.add_argument("--grad_clip",         default=1.0,                   type=float)
    parser.add_argument("--num_samples",       default=30,                    type=int)
    parser.add_argument("--out_dir",           default="generated",           type=str)
    parser.add_argument("--device",            default="cuda", choices=["cuda","cpu"])
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    raw = pd.read_csv(args.data_path, header=None).values

    samples = []
    for i in range(args.num_raw_samples):
        arr = raw[i*args.seq_len:(i+1)*args.seq_len]
        samples.append(torch.tensor((arr-6.0)/19.0, dtype=torch.float32).to(device))
    def get_case(idxs):
        return torch.stack([samples[i] for i in idxs], dim=0)
    # cases = {'case1':[0,1],'case2':[1,2],'case3':[0,2]}
    args.feat_dim = raw.shape[1]
    cases = {'case1':[0,1],'case2':[1,2],'case3':[0,2]}

    methods = [
        ("GAN",       train_gan),
        ("AE",        train_ae),
        ("VAE",       train_vae),
        ("TimeGAN",   train_timegan),
        ("Diffusion", train_diffusion),
    ]

    total_tasks = len(cases) * len(methods)

    with tqdm(total=total_tasks, desc="Overall generation", unit="task") as pbar:
        for case_name, idxs in cases.items():
            data = get_case(idxs)
            for method_name, func in methods:
                pbar.set_postfix(case=case_name, method=method_name)
                func(data, case_name)
                pbar.update(1)

