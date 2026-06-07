#!/usr/bin/env python3
"""
认知拓扑涌现与导航约束实验（改进版）
改进：
1. 导航步数增至2000，提升单次信度
2. 增加覆盖节点数（Coverage）作为行为互补指标
3. 随机初始化对照扩展至50种子
4. 新增KNN环境图对照（等度，消除不对称性质疑）
5. 干预实验预存ΔM，多中介分析预先声明
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import community as community_louvain
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, time, warnings
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ======================== 配置 ========================
class Cfg:
    N_NODES = 1000
    EMBEDDING_DIM = 50
    SEEDS_EXP1 = list(range(50))          # 主实验50种子
    SEEDS_CTRL = list(range(100, 150))    # 随机初始化对照50种子（独立）
    SEEDS_SWEEP = list(range(20))         # 参数扫描种子
    TRAIN_STEPS = 200_000
    RECORD_INTERVAL = 20_000
    COOCCUR_MIN = 2
    COOCCUR_MAX = 5
    LATERAL_K = 10
    ALPHA = 0.01
    BETA = 0.01
    R_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    T_VALUES = [0.5, 1.0, 2.0]           # 温度扫描
    Q_THRESHOLDS = [80, 85, 90, 95]       # 环境图阈值
    KNN_K = 5
    NAV_STEPS = 2000                      # 改进：2000步
    NAV_REPEATS = 20
    EPSILON = 0.05
    REWIRE_FRACS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    VEC_CLIP = 10.0
    OUTPUT_DIR = "output_v2"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = Cfg()

# ======================== 环境与学习规则 ========================
class SemanticEnvironment:
    def __init__(self, seed, T=0.5):
        self.rng = np.random.default_rng(seed)
        self.vectors = self.rng.normal(0, 1, (cfg.N_NODES, cfg.EMBEDDING_DIM))
        self.T = T

    def generate(self):
        anchor = self.rng.integers(0, cfg.N_NODES)
        diff = np.sum((self.vectors - self.vectors[anchor])**2, axis=1)
        diff[anchor] = np.inf
        probs = np.exp(-diff / self.T)
        probs /= probs.sum()
        n_co = self.rng.integers(cfg.COOCCUR_MIN, cfg.COOCCUR_MAX+1)
        companions = self.rng.choice(cfg.N_NODES, size=n_co, replace=False, p=probs)
        cooccur = np.concatenate([[anchor], companions])
        mask = np.ones(cfg.N_NODES, dtype=bool)
        mask[cooccur] = False
        n_lat = min(cfg.LATERAL_K, mask.sum())
        lateral = self.rng.choice(np.where(mask)[0], size=n_lat, replace=False)
        return cooccur, lateral

def train_vectors(seed, alpha=cfg.ALPHA, beta=cfg.BETA, T=0.5):
    rng = np.random.default_rng(seed)
    env = SemanticEnvironment(seed, T)
    vectors = rng.normal(0, 0.01, (cfg.N_NODES, cfg.EMBEDDING_DIM))
    history = []
    for step in range(1, cfg.TRAIN_STEPS + 1):
        co, lat = env.generate()
        # 赫布学习
        for i_idx in range(len(co)):
            i_node = co[i_idx]
            for j_idx in range(i_idx+1, len(co)):
                j_node = co[j_idx]
                diff = vectors[j_node] - vectors[i_node]
                vectors[i_node] += alpha * diff
                vectors[j_node] -= alpha * diff
        # 侧抑制
        for i_node in co:
            for k_node in lat:
                diff = vectors[k_node] - vectors[i_node]
                vectors[i_node] -= beta * diff
        np.clip(vectors, -cfg.VEC_CLIP, cfg.VEC_CLIP, out=vectors)
        if np.any(np.isnan(vectors)):
            raise RuntimeError(f"NaN detected at step {step}, seed {seed}")
        if step % cfg.RECORD_INTERVAL == 0:
            _, G_temp = build_knn(vectors)
            S_val = largest_component_size(G_temp)
            Q_val, _ = modularity_fixed(G_temp)
            history.append((step, S_val, Q_val))
    _, G_final = build_knn(vectors)
    return vectors, G_final, history, env

# ======================== 网络指标 ========================
def build_knn(vectors, k=cfg.KNN_K):
    if np.any(~np.isfinite(vectors)):
        raise ValueError("vectors contain non-finite values")
    sim = cosine_similarity(vectors)
    n = sim.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        s = sim[i].copy(); s[i] = -np.inf
        top_k = np.argpartition(-s, k)[:k]
        adj[i, top_k] = s[top_k]
    G = np.maximum(adj, adj.T)
    return adj, G

def largest_component_size(G):
    n_comp, labels = connected_components(csr_matrix(G), directed=False, return_labels=True)
    return np.max(np.bincount(labels)) / len(G) if n_comp > 0 else 0.0

def clustering_coefficient(G):
    n = G.shape[0]
    total = 0.0
    for i in range(n):
        neighbors = np.where(G[i] > 0)[0]
        k = len(neighbors)
        if k < 2: continue
        sub = G[np.ix_(neighbors, neighbors)]
        e = np.sum(sub > 0) / 2
        total += (2 * e) / (k * (k - 1))
    return total / n

def avg_shortest_path(G):
    dist = shortest_path(csr_matrix(G), directed=False, unweighted=True)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    return np.mean(finite) if len(finite) > 0 else np.inf

def modularity_fixed(G, seed=42):
    if np.sum(G) == 0:
        return 0.0, {}
    G_nx = nx.from_numpy_array(G)
    try:
        part = community_louvain.best_partition(G_nx, random_state=seed)
        Q = community_louvain.modularity(part, G_nx)
    except (ValueError, ZeroDivisionError):
        Q = 0.0
        part = {}
    return Q, part

def module_cohesion(G, labels):
    intra, inter = [], []
    n = G.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if G[i, j] > 0:
                if labels[i] == labels[j]:
                    intra.append(G[i, j])
                else:
                    inter.append(G[i, j])
    if len(inter) == 0:
        return 1.0 if len(intra) > 0 else 0.0
    return np.mean(intra) / np.mean(inter) if intra else 0.0

def vector_shuffle_null(vectors, G, n_perm=100):
    n = vectors.shape[0]
    Cs, Ls = [], []
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        _, G_null = build_knn(vectors[perm])
        Cs.append(clustering_coefficient(G_null))
        Ls.append(avg_shortest_path(G_null))
    return np.mean(Cs), np.mean(Ls)

# ======================== 导航与干预 ========================
class Navigator:
    def __init__(self, G, epsilon=cfg.EPSILON):
        self.G = G
        self.n = G.shape[0]
        self.eps = epsilon
        self.best = {}
        for i in range(self.n):
            w = G[i]
            self.best[i] = np.argmax(w) if w.sum() > 0 else i

    def run(self, steps, start=None):
        if start is None: start = np.random.randint(0, self.n)
        traj = [start]
        cur = start
        rng = np.random.default_rng()
        for _ in range(steps):
            if rng.random() < self.eps:
                cur = rng.integers(0, self.n)
            else:
                cur = self.best[cur]
            traj.append(cur)
        return np.array(traj)

def navigation_entropy_and_coverage(G, epsilon=cfg.EPSILON, steps=cfg.NAV_STEPS, repeats=cfg.NAV_REPEATS):
    """返回平均熵、熵标准差、原始熵列表、平均覆盖节点数"""
    nav = Navigator(G, epsilon)
    ents = []
    covs = []
    for _ in range(repeats):
        traj = nav.run(steps)
        unique_nodes = len(np.unique(traj))
        covs.append(unique_nodes)
        _, cnts = np.unique(traj, return_counts=True)
        ents.append(entropy(cnts))
    return np.mean(ents), np.std(ents), ents, np.mean(covs)

def rewire_preserve_degrees(G, fraction):
    Gnew = G.copy()
    n = G.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if Gnew[i, j] > 0]
    np.random.shuffle(edges)
    n_rewire = int(len(edges) * fraction)
    for _ in range(n_rewire):
        if len(edges) < 2: break
        (a, b), (c, d) = edges.pop(), edges.pop()
        if a == d or c == b or Gnew[a, d] > 0 or Gnew[c, b] > 0:
            continue
        w1, w2 = Gnew[a, b], Gnew[c, d]
        Gnew[a, b] = Gnew[b, a] = 0
        Gnew[c, d] = Gnew[d, c] = 0
        Gnew[a, d] = Gnew[d, a] = w1
        Gnew[c, b] = Gnew[b, c] = w2
    return Gnew

# ======================== 统计检验 ========================
def perm_test_paired(diffs, n_perm=10000, alternative='greater'):
    diffs = np.array(diffs)
    obs = np.mean(diffs)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        if alternative == 'greater':
            count += (np.mean(diffs * signs) >= obs)
        else:
            count += (np.mean(diffs * signs) <= obs)
    return (count + 1) / (n_perm + 1)

def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    data = np.array(data)
    boots = np.random.choice(data, size=(n_boot, len(data)), replace=True)
    means = np.mean(boots, axis=1)
    return np.percentile(means, 100*alpha/2), np.percentile(means, 100*(1-alpha/2))

def icc1(measurements):
    n, k = measurements.shape
    if k < 2: return 1.0
    mean_subj = np.mean(measurements, axis=1)
    grand_mean = np.mean(measurements)
    ms_b = np.sum((mean_subj - grand_mean)**2) * k / (n - 1) if n > 1 else 0
    ms_w = np.sum((measurements - mean_subj.reshape(-1, 1))**2) / (n * (k - 1))
    return (ms_b - ms_w) / (ms_b + (k - 1) * ms_w) if ms_b > ms_w else 0.0

# ======================== 实验1：认知网络涌现 ========================
def run_experiment1():
    print("="*60, flush=True)
    print("实验1：认知网络的涌现", flush=True)
    print("="*60, flush=True)
    results = []
    n_seeds = len(cfg.SEEDS_EXP1)
    t_start = time.time()

    for idx, seed in enumerate(cfg.SEEDS_EXP1):
        t_seed_start = time.time()
        try:
            vec, G, hist, env = train_vectors(seed=seed, T=0.5)
        except Exception as e:
            print(f"  [种子 {seed}] 训练失败: {e}", flush=True)
            continue
        S_val = largest_component_size(G)
        C_val = clustering_coefficient(G)
        L_val = avg_shortest_path(G)
        Q_val, part = modularity_fixed(G)
        cohesion_val = module_cohesion(G, part)
        Cnull, Lnull = vector_shuffle_null(vec, G)
        sigma_val = (C_val / Cnull) / (L_val / Lnull) if Cnull and Lnull else 0.0

        # 环境共现矩阵模块度 (阈值法)
        cooc_mat = np.zeros((cfg.N_NODES, cfg.N_NODES))
        for _ in range(50000):
            co, _ = env.generate()
            for i in range(len(co)):
                for j in range(i+1, len(co)):
                    cooc_mat[co[i], co[j]] += 1
                    cooc_mat[co[j], co[i]] += 1
        Q_env_dict = {}
        if np.any(cooc_mat > 0):
            for thresh_perc in cfg.Q_THRESHOLDS:
                threshold = np.percentile(cooc_mat[cooc_mat > 0], thresh_perc)
                env_G = (cooc_mat > threshold).astype(float)
                Q_env, _ = modularity_fixed(env_G)
                Q_env_dict[thresh_perc] = Q_env
        else:
            for thresh_perc in cfg.Q_THRESHOLDS:
                Q_env_dict[thresh_perc] = 0.0

        # 改进：KNN环境图对照（等度）
        env_knn_G, _ = build_knn(env.vectors, k=cfg.KNN_K)
        Q_env_knn, _ = modularity_fixed(env_knn_G)

        results.append({
            'seed': seed,
            'S': S_val, 'C': C_val, 'L': L_val, 'Q': Q_val,
            'cohesion': cohesion_val, 'sigma': sigma_val,
            'Q_env': Q_env_dict, 'Q_env_knn': Q_env_knn,
            'history': hist,
            'vectors': vec, 'G': G, 'env': env
        })

        elapsed_total = time.time() - t_start
        elapsed_this = time.time() - t_seed_start
        done = idx + 1
        remain = n_seeds - done
        if done > 0:
            avg_per_seed = elapsed_total / done
            eta = avg_per_seed * remain
            print(f"  [种子 {done:3d}/{n_seeds}] 本次耗时 {elapsed_this:.0f}s | "
                  f"总已用 {elapsed_total/60:.1f}min | 预估剩余 {eta/60:.1f}min", flush=True)

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No successful experiments.")

    # 随机初始化对照（扩展至50种子）
    print("\n随机初始化对照（50种子）...", flush=True)
    ctrl_S, ctrl_C, ctrl_Q = [], [], []
    for seed in cfg.SEEDS_CTRL:
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 0.01, (cfg.N_NODES, cfg.EMBEDDING_DIM))
        _, G = build_knn(vec)
        ctrl_S.append(largest_component_size(G))
        ctrl_C.append(clustering_coefficient(G))
        Q_val, _ = modularity_fixed(G)
        ctrl_Q.append(Q_val)
    ctrl_S_median = np.median(ctrl_S)
    ctrl_S_ci = bootstrap_ci(ctrl_S)
    ctrl_Q_median = np.median(ctrl_Q)

    # 收敛曲线
    steps = [h[0] for h in results[0]['history']]
    S_hist = np.array([[h[1] for h in r['history']] for r in results])
    Q_hist = np.array([[h[2] for h in r['history']] for r in results])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, np.mean(S_hist, axis=0))
    plt.fill_between(steps, np.percentile(S_hist, 2.5, axis=0),
                     np.percentile(S_hist, 97.5, axis=0), alpha=0.3)
    plt.xlabel('Training steps'); plt.ylabel('Largest component S'); plt.title('Convergence of S')
    plt.subplot(1, 2, 2)
    plt.plot(steps, np.mean(Q_hist, axis=0))
    plt.fill_between(steps, np.percentile(Q_hist, 2.5, axis=0),
                     np.percentile(Q_hist, 97.5, axis=0), alpha=0.3)
    plt.xlabel('Training steps'); plt.ylabel('Modularity Q'); plt.title('Convergence of Q')
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/convergence.png"); plt.close()

    # 小世界检验
    sigma_vals = df['sigma'].values
    median_sigma_obs = np.median(sigma_vals)
    n_null_samples = 1000
    null_medians = []
    for _ in range(n_null_samples):
        null_sample = []
        for r in results:
            Cnull, Lnull = vector_shuffle_null(r['vectors'], r['G'], 1)
            sigma_null = (r['C'] / Cnull) / (r['L'] / Lnull) if Cnull and Lnull else 1.0
            null_sample.append(sigma_null)
        null_medians.append(np.median(null_sample))
    p_sigma = max(np.mean(np.array(null_medians) >= median_sigma_obs), 1e-5)

    # 环境模块度鲁棒性
    Q_vals = df['Q'].values
    q_thresh_results = {}
    for thresh_perc in cfg.Q_THRESHOLDS:
        Q_env_vals = np.array([r['Q_env'][thresh_perc] for r in results])
        diffs = Q_vals - Q_env_vals
        p_val = perm_test_paired(diffs, n_perm=10000, alternative='greater')
        q_thresh_results[thresh_perc] = {'median_diff': np.median(diffs), 'p_value': p_val}
    # KNN环境图对照
    Q_env_knn_vals = np.array([r['Q_env_knn'] for r in results])
    diffs_knn = Q_vals - Q_env_knn_vals
    p_knn = perm_test_paired(diffs_knn, n_perm=10000, alternative='greater')

    print("\n环境模块度鲁棒性:", flush=True)
    print("阈值分位数  中位差   p值", flush=True)
    for th, res in q_thresh_results.items():
        print(f"  {th}%        {res['median_diff']:.4f}   {res['p_value']:.4f}", flush=True)
    print(f"KNN环境图对照: Q中位差 = {np.median(diffs_knn):.4f}, p = {p_knn:.4f}", flush=True)

    # 参数扫描 (r, T)
    sweep_r = []
    for r in cfg.R_VALUES:
        alpha = r * cfg.BETA
        for s in cfg.SEEDS_SWEEP:
            try:
                vec, G, _, _ = train_vectors(s, alpha, cfg.BETA, T=0.5, show_progress=False)
                S_val = largest_component_size(G)
                sweep_r.append({'r': r, 'seed': s, 'S': S_val})
            except: continue
    df_r = pd.DataFrame(sweep_r)

    sweep_T = []
    for T_val in cfg.T_VALUES:
        for s in cfg.SEEDS_SWEEP[:10]:
            try:
                vec, G, _, _ = train_vectors(s, cfg.ALPHA, cfg.BETA, T_val, show_progress=False)
                S_val = largest_component_size(G)
                C_val = clustering_coefficient(G)
                Q_val, _ = modularity_fixed(G)
                sweep_T.append({'T': T_val, 'seed': s, 'S': S_val, 'C': C_val, 'Q': Q_val})
            except: continue
    df_T = pd.DataFrame(sweep_T)

    # 统计摘要
    S_median = float(np.median(df['S']))
    S_ci = bootstrap_ci(df['S'])
    stats = {
        'S_median': S_median,
        'S_ci_low': float(S_ci[0]), 'S_ci_high': float(S_ci[1]),
        'sigma_median': float(median_sigma_obs),
        'sigma_ci_low': float(bootstrap_ci(sigma_vals)[0]),
        'sigma_ci_high': float(bootstrap_ci(sigma_vals)[1]),
        'p_sigma_greater_1': float(p_sigma),
        'Q_median': float(np.median(Q_vals)),
        'cohesion_median': float(np.median(df['cohesion'])),
        'ctrl_S_median': ctrl_S_median,
        'ctrl_S_ci': (float(ctrl_S_ci[0]), float(ctrl_S_ci[1])),
        'ctrl_Q_median': ctrl_Q_median,
        'Q_env_robustness': q_thresh_results,
        'Q_env_knn_diff': float(np.median(diffs_knn)),
        'p_Q_env_knn': p_knn
    }
    print("\n关键统计结果:", flush=True)
    for k, v in stats.items():
        if k not in ['Q_env_robustness']:
            print(f"  {k}: {v}", flush=True)
    with open(f"{cfg.OUTPUT_DIR}/exp1_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    df.to_pickle(f"{cfg.OUTPUT_DIR}/exp1_results.pkl")
    df_r.to_pickle(f"{cfg.OUTPUT_DIR}/sweep_r.pkl")
    df_T.to_pickle(f"{cfg.OUTPUT_DIR}/sweep_T.pkl")
    return results, stats, df_r, df_T

# ======================== 实验2：导航约束 ========================
def run_experiment2(networks):
    print("\n" + "="*60, flush=True)
    print("实验2：导航行为的结构约束", flush=True)
    print("="*60, flush=True)
    base = []
    n_nets = min(len(networks), 20)
    for idx, net in enumerate(networks[:n_nets]):
        G = net['G']
        C_val = clustering_coefficient(G)
        Q_val, part = modularity_fixed(G)
        coh_val = module_cohesion(G, part)
        H_mean, H_std, raw_ents, cov_mean = navigation_entropy_and_coverage(G)
        base.append({
            'net_id': idx, 'C': C_val, 'Q': Q_val, 'cohesion': coh_val,
            'H_mean': H_mean, 'H_std': H_std, 'raw_ents': raw_ents,
            'coverage_mean': cov_mean, 'G': G
        })
        print(f"  导航分析: 网络 {idx+1}/{n_nets} 完成 (熵={H_mean:.3f}, 覆盖={cov_mean:.1f})", flush=True)
    df_base = pd.DataFrame(base)

    # 信度与个体差异
    ent_matrix = np.array([row for row in df_base['raw_ents']])
    icc_val = icc1(ent_matrix)
    print(f"\n导航熵 ICC(1) = {icc_val:.4f}", flush=True)
    # 覆盖节点数的信度 (近似)
    cov_matrix = np.full((20, 20), np.nan)  # 这里简化，若需要可存储原始覆盖数据
    print("补充分析：覆盖节点数 (Coverage) 与拓扑指标的相关")
    print(f"  C vs Coverage: r = {np.corrcoef(df_base['C'], df_base['coverage_mean'])[0,1]:.3f}")
    print(f"  M vs Coverage: r = {np.corrcoef(df_base['cohesion'], df_base['coverage_mean'])[0,1]:.3f}")

    # 回归分析 (同前，加入Coverage作为因变量检查)
    X = df_base[['C', 'cohesion']]
    y = df_base['H_mean']
    reg = LinearRegression().fit(X, y)
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()
    print("\n回归结果 (H ~ C + cohesion):", flush=True)
    print(ols_model.summary().tables[1], flush=True)

    # 干预实验 (预存ΔM)
    inter = []
    for idx, net in enumerate(networks[:20]):
        G_orig = net['G']
        C0 = clustering_coefficient(G_orig)
        Q0, part = modularity_fixed(G_orig)
        coh0 = module_cohesion(G_orig, part)
        H0, _, _, _ = navigation_entropy_and_coverage(G_orig)
        for frac in cfg.REWIRE_FRACS:
            G_int = rewire_preserve_degrees(G_orig, frac)
            C_int = clustering_coefficient(G_int)
            Q_int, part_int = modularity_fixed(G_int)
            coh_int = module_cohesion(G_int, part_int)
            H_int, _, _, _ = navigation_entropy_and_coverage(G_int)
            inter.append({
                'net_id': idx, 'frac': frac,
                'C_orig': C0, 'C_int': C_int,
                'M_orig': coh0, 'M_int': coh_int,
                'H_orig': H0, 'H_int': H_int,
                'dC': C_int - C0, 'dM': coh_int - coh0, 'dH': H_int - H0
            })
        print(f"  干预实验: 网络 {idx+1}/20 完成 (7个重连比例, ΔM已记录)", flush=True)
    df_int = pd.DataFrame(inter)

    # 剂量-反应
    dose = df_int.groupby('frac')[['dC', 'dH']].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.plot(dose['frac'], -dose['dC'], 'o-', label='-ΔC')
    plt.plot(dose['frac'], dose['dH'], 's-', label='ΔH')
    plt.xlabel('Rewiring fraction'); plt.ylabel('Change')
    plt.legend(); plt.title('Dose-response of rewiring')
    plt.savefig(f"{cfg.OUTPUT_DIR}/dose_response.png"); plt.close()

    # 中介分析：ΔC路径（预先声明多中介）
    a_model = sm.OLS(df_int['dC'], sm.add_constant(df_int['frac'])).fit()
    bc_model = sm.OLS(df_int['dH'], sm.add_constant(df_int[['frac', 'dC']])).fit()
    indirect = a_model.params['frac'] * bc_model.params['dC']
    p_dC = bc_model.pvalues['dC']
    print(f"\n中介分析 (frac -> dC -> dH):", flush=True)
    print(f"  a (frac→dC) = {a_model.params['frac']:.4f}", flush=True)
    print(f"  b (dC→dH)   = {bc_model.params['dC']:.4f}", flush=True)
    print(f"  间接效应 = {indirect:.4f}, dC的p值 = {p_dC:.4f}", flush=True)

    # 多中介：frac -> dC + dM -> dH
    print("\n多中介分析 (预先声明):", flush=True)
    X_multi = sm.add_constant(df_int[['frac', 'dC', 'dM']])
    multi_model = sm.OLS(df_int['dH'], X_multi).fit()
    print(multi_model.summary().tables[1], flush=True)

    df_base.to_pickle(f"{cfg.OUTPUT_DIR}/exp2_base.pkl")
    df_int.to_pickle(f"{cfg.OUTPUT_DIR}/exp2_intervention.pkl")
    return df_base, df_int, ols_model

# ======================== 主入口 ========================
def main():
    t0 = time.time()
    print("\n认知拓扑涌现实验 (改进版) — 开始运行\n", flush=True)
    results_exp1, stats_exp1, sweep_r, sweep_T = run_experiment1()
    base, inter, reg_model = run_experiment2(results_exp1)
    elapsed = time.time() - t0
    print(f"\n全部实验完成，总耗时 {elapsed/60:.1f} 分钟 ({elapsed/3600:.2f} 小时)", flush=True)
    print(f"输出文件保存在 '{cfg.OUTPUT_DIR}' 目录下", flush=True)

if __name__ == "__main__":
    main()