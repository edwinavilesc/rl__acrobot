"""DQN reproducible para U3 Taller 2 — Acrobot-v1.

Flujo: preflight -> smoke/full training -> greedy evaluation -> artifacts.
"""
from dataclasses import dataclass, asdict, replace
from collections import deque
from pathlib import Path
import json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    env_id: str = "Acrobot-v1"
    episodes: int = 1200
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    target_update_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    hidden: int = 128
    max_grad_norm: float = 10.0
    seed: int = 42
    eval_episodes: int = 100

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
    def push(self, s, a, r, s2, terminated):
        self.data.append((np.asarray(s,np.float32), int(a), float(r), np.asarray(s2,np.float32), float(terminated)))
    def sample(self, batch_size):
        batch = random.sample(self.data,batch_size)
        return tuple(map(np.asarray,zip(*batch)))
    def __len__(self):
        return len(self.data)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def epsilon_by_step(step,cfg):
    frac=min(1.0,step/cfg.epsilon_decay_steps)
    return cfg.epsilon_start+frac*(cfg.epsilon_end-cfg.epsilon_start)

def choose_action(obs,q_net,n_actions,eps):
    if random.random()<eps: return random.randrange(n_actions)
    x=torch.as_tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    with torch.no_grad(): return int(q_net(x).argmax(dim=1).item())

def optimize_step(q_net,target_net,optimizer,buffer,cfg):
    if len(buffer)<max(cfg.batch_size,cfg.learning_starts): return None
    s,a,r,s2,terminated=buffer.sample(cfg.batch_size)
    s=torch.as_tensor(s,dtype=torch.float32,device=DEVICE)
    a=torch.as_tensor(a,dtype=torch.long,device=DEVICE).unsqueeze(1)
    r=torch.as_tensor(r,dtype=torch.float32,device=DEVICE).unsqueeze(1)
    s2=torch.as_tensor(s2,dtype=torch.float32,device=DEVICE)
    terminated=torch.as_tensor(terminated,dtype=torch.float32,device=DEVICE).unsqueeze(1)
    current_q=q_net(s).gather(1,a)
    with torch.no_grad():
        next_q=target_net(s2).max(dim=1,keepdim=True).values
        target_q=r+cfg.gamma*next_q*(1-terminated)
    loss=nn.functional.smooth_l1_loss(current_q,target_q)
    optimizer.zero_grad(); loss.backward()
    grad_norm=float(nn.utils.clip_grad_norm_(q_net.parameters(),cfg.max_grad_norm))
    optimizer.step()
    return float(loss.item()), grad_norm

def train_dqn(cfg, artifact_dir=Path("artifacts")):
    artifact_dir.mkdir(exist_ok=True)
    seed_everything(cfg.seed)
    env=gym.make(cfg.env_id); env.action_space.seed(cfg.seed)
    obs_dim=int(np.prod(env.observation_space.shape)); n_actions=env.action_space.n
    q=QNetwork(obs_dim,n_actions,cfg.hidden).to(DEVICE)
    tgt=QNetwork(obs_dim,n_actions,cfg.hidden).to(DEVICE); tgt.load_state_dict(q.state_dict()); tgt.eval()
    opt=optim.Adam(q.parameters(),lr=cfg.lr); buf=ReplayBuffer(cfg.buffer_size)
    h={"returns":[],"lengths":[],"losses":[],"grad_norms":[],"epsilons":[],"terminated":[],"truncated":[]}
    step=0; start=time.time()
    for ep in range(cfg.episodes):
        obs,_=env.reset(seed=cfg.seed+ep); total=0.0; nstep=0
        while True:
            eps=epsilon_by_step(step,cfg); a=choose_action(obs,q,n_actions,eps)
            nxt,r,terminated,truncated,_=env.step(a)
            buf.push(obs,a,r,nxt,terminated)
            out=optimize_step(q,tgt,opt,buf,cfg)
            if out is not None:
                loss,gn=out; h["losses"].append(loss); h["grad_norms"].append(gn)
            obs=nxt; total+=r; nstep+=1; step+=1
            if step % cfg.target_update_steps==0: tgt.load_state_dict(q.state_dict())
            if terminated or truncated: break
        h["returns"].append(total); h["lengths"].append(nstep); h["epsilons"].append(eps)
        h["terminated"].append(bool(terminated)); h["truncated"].append(bool(truncated))
        if (ep+1)%100==0: print(f"ep={ep+1:4d} mean100={np.mean(h['returns'][-100:]):8.2f} eps={eps:.3f}")
    h["seconds"]=time.time()-start; h["config"]=asdict(cfg); env.close()
    torch.save(q.state_dict(),artifact_dir/f"dqn_acrobot_{cfg.episodes}ep_state_dict.pt")
    (artifact_dir/f"training_history_{cfg.episodes}ep.json").write_text(json.dumps(h,indent=2))
    return q,h

def evaluate(q_net,cfg,n=None,seed_start=10_000):
    n=n or cfg.eval_episodes; env=gym.make(cfg.env_id); rets=[]; lens=[]; terms=[]
    for seed in range(seed_start,seed_start+n):
        obs,_=env.reset(seed=seed); total=0.0; steps=0
        while True:
            a=choose_action(obs,q_net,env.action_space.n,0.0)
            obs,r,terminated,truncated,_=env.step(a); total+=r; steps+=1
            if terminated or truncated:
                rets.append(total); lens.append(steps); terms.append(bool(terminated)); break
    env.close()
    return {"n":len(rets),"mean_return":float(np.mean(rets)),"median_return":float(np.median(rets)),
            "sd_return":float(np.std(rets,ddof=1)),"mean_steps":float(np.mean(lens)),
            "terminal_rate":float(np.mean(terms)),"returns":rets,"lengths":lens}

if __name__ == "__main__":
    assert "Acrobot-v1" in gym.envs.registry
    mode="smoke"
    cfg=Config(); cfg=replace(cfg,episodes=(10 if mode=="smoke" else cfg.episodes))
    print("device:",DEVICE,"mode:",mode,"episodes:",cfg.episodes)
    q,h=train_dqn(cfg)
    m=evaluate(q,cfg,n=(10 if mode=="smoke" else cfg.eval_episodes))
    print({k:v for k,v in m.items() if k not in ("returns","lengths")})
