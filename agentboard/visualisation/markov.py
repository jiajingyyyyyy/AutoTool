import json
import numpy as np
import os
from collections import defaultdict, Counter
from scipy.stats import chi2, entropy
from itertools import groupby
import matplotlib.pyplot as plt

# ============================================================================
# 数据加载与预处理
# ============================================================================

def load_all_trajectories(directory_path):
    """加载目录下所有轨迹文件并合并"""
    all_sequences = []
    file_list = [
        os.path.abspath(os.path.join(directory_path, f)) 
        for f in os.listdir(directory_path) 
        if f.endswith('.json')
    ]
    
    print(f"Found {len(file_list)} trajectory files in {directory_path}")
    
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sequences = data.get('sequences', [])
            all_sequences.extend(sequences)
            print(f"  Loaded {len(sequences)} sequences from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    return all_sequences


def extract_tool_sequences(sequences, filter_tools=None, remove_consecutive_duplicates=False):
    """
    提取工具调用序列
    :param sequences: 原始序列数据
    :param filter_tools: 要过滤掉的工具列表
    :param remove_consecutive_duplicates: 是否去除连续重复的工具调用
    """
    if filter_tools is None:
        filter_tools = []
    
    tool_sequences = []
    for seq in sequences:
        if 'steps' in seq:
            tools = [
                step['action']['parsed_content']['tool_name'] 
                for step in seq['steps'] 
                if 'action' in step 
                and 'parsed_content' in step['action'] 
                and 'tool_name' in step['action']['parsed_content']
                and step['action']['parsed_content']['tool_name'] not in filter_tools
            ]
            
            # 使用 groupby 去除连续重复
            if remove_consecutive_duplicates and tools:
                tools = [key for key, _ in groupby(tools)]
            
            if tools:
                tool_sequences.append(tools)
    
    return tool_sequences


# ============================================================================
# 马尔可夫矩阵计算
# ============================================================================

def calculate_0th_order_matrix(tool_sequences):
    """计算 0 阶马尔可夫矩阵（工具出现频率）"""
    tool_counts = Counter()
    total_count = 0
    
    for sequence in tool_sequences:
        for tool in sequence:
            tool_counts[tool] += 1
            total_count += 1
    
    tool_probs = {tool: count / total_count for tool, count in tool_counts.items()}
    return tool_probs, tool_counts


def calculate_kth_order_matrix(tool_sequences, k):
    """
    计算 k 阶马尔可夫转移矩阵
    :param k: 马尔可夫阶数（1 或 2）
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    total_transitions = defaultdict(int)
    
    for sequence in tool_sequences:
        for i in range(len(sequence) - k):
            if k == 1:
                context = sequence[i]
            else:
                context = tuple(sequence[i:i+k])
            next_tool = sequence[i+k]
            transition_counts[context][next_tool] += 1
            total_transitions[context] += 1
    
    # 归一化为概率
    transition_probs = {}
    for context, next_tools in transition_counts.items():
        transition_probs[context] = {
            next_tool: count / total_transitions[context]
            for next_tool, count in next_tools.items()
        }
    
    return transition_probs, transition_counts, total_transitions


# ============================================================================
# 熵计算
# ============================================================================

def calculate_entropy(prob_dist):
    """计算熵（安全处理零概率）"""
    probs = np.array(list(prob_dist.values()), dtype=float)
    probs = probs[probs > 0]  # 移除零概率
    if len(probs) == 0:
        return 0
    return entropy(probs, base=2)


def calculate_conditional_entropy(transition_probs, prior_probs):
    """计算条件熵 H(X|Y)"""
    conditional_entropy = 0
    for context, next_tools in transition_probs.items():
        context_prob = prior_probs.get(context, 0)
        if context_prob > 0:
            next_probs = np.array(list(next_tools.values()), dtype=float)
            next_probs = next_probs[next_probs > 0]
            if len(next_probs) > 0:
                conditional_entropy += context_prob * entropy(next_probs, base=2)
    return conditional_entropy


def analyze_entropy_reduction(tool_sequences):
    """计算 0/1/2 阶熵及熵减"""
    # 0 阶
    tool_probs, _ = calculate_0th_order_matrix(tool_sequences)
    H0 = calculate_entropy(tool_probs)
    
    # 1 阶
    transition_probs_1, _, _ = calculate_kth_order_matrix(tool_sequences, k=1)
    H1 = calculate_conditional_entropy(transition_probs_1, tool_probs)
    
    # 2 阶：需要计算 2 元组的先验概率
    transition_probs_2, _, total_transitions_2 = calculate_kth_order_matrix(tool_sequences, k=2)
    context_probs = {}
    total_contexts = sum(total_transitions_2.values())
    if total_contexts > 0:
        for context, count in total_transitions_2.items():
            context_probs[context] = count / total_contexts
        H2 = calculate_conditional_entropy(transition_probs_2, context_probs)
    else:
        H2 = H1
    
    # 熵减
    delta_H1 = H0 - H1  # 1阶 vs 0阶
    delta_H2_vs_1 = H1 - H2  # 2阶 vs 1阶
    delta_H2_vs_0 = H0 - H2  # 2阶 vs 0阶（新增）
    
    # 相对熵减
    relative_red_1 = (delta_H1 / H0) * 100 if H0 > 0 else 0
    relative_red_2_vs_1 = (delta_H2_vs_1 / H1) * 100 if H1 > 0 else 0
    relative_red_2_vs_0 = (delta_H2_vs_0 / H0) * 100 if H0 > 0 else 0
    
    return {
        'H0': H0, 'H1': H1, 'H2': H2,
        'delta_H1': delta_H1,
        'delta_H2_vs_1': delta_H2_vs_1,
        'delta_H2_vs_0': delta_H2_vs_0,  # 新增
        'relative_red_1': relative_red_1,
        'relative_red_2_vs_1': relative_red_2_vs_1,
        'relative_red_2_vs_0': relative_red_2_vs_0  # 新增
    }


# ============================================================================
# 似然比检验（LRT）- 修正版
# ============================================================================

def likelihood_ratio_test(tool_sequences, k_high, k_low=0):
    """
    k_high 阶马尔可夫模型 vs k_low 阶模型的似然比检验
    :param k_high: 高阶模型阶数
    :param k_low: 低阶模型阶数（默认 0）
    """
    if k_high <= k_low:
        return None, None, None
    
    # 统计频次
    marginal_counts = Counter()
    
    # 低阶模型
    if k_low == 0:
        # 0 阶：边际分布
        low_counts = defaultdict(int)
        for seq in tool_sequences:
            for tool in seq:
                marginal_counts[tool] += 1
                low_counts[tool] += 1
    else:
        # k_low 阶转移
        low_counts = defaultdict(lambda: defaultdict(int))
        for seq in tool_sequences:
            for tool in seq:
                marginal_counts[tool] += 1
            for i in range(len(seq) - k_low):
                if k_low == 1:
                    context = seq[i]
                else:
                    context = tuple(seq[i:i+k_low])
                next_tool = seq[i+k_low]
                low_counts[context][next_tool] += 1
    
    # 高阶模型
    high_counts = defaultdict(lambda: defaultdict(int))
    for seq in tool_sequences:
        for i in range(len(seq) - k_high):
            if k_high == 1:
                context = seq[i]
            else:
                context = tuple(seq[i:i+k_high])
            next_tool = seq[i+k_high]
            high_counts[context][next_tool] += 1
    
    # 计算对数似然 - 低阶模型
    log_L_low = 0
    if k_low == 0:
        total = sum(low_counts.values())
        for count in low_counts.values():
            if count > 0:
                p = count / total
                log_L_low += count * np.log(p)
    else:
        for context, next_tools in low_counts.items():
            context_total = sum(next_tools.values())
            for count in next_tools.values():
                if count > 0:
                    p = count / context_total
                    log_L_low += count * np.log(p)
    
    # 计算对数似然 - 高阶模型
    log_L_high = 0
    for context, next_tools in high_counts.items():
        context_total = sum(next_tools.values())
        for count in next_tools.values():
            if count > 0:
                p = count / context_total
                log_L_high += count * np.log(p)
    
    # G² 统计量
    G2 = 2 * (log_L_high - log_L_low)
    
    # 自由度（修正版）
    T = len(marginal_counts)
    if k_low == 0:
        # k 阶 vs 0 阶
        # 实际观察到的上下文数量（而非理论最大值）
        num_contexts_high = len(high_counts)
        df = num_contexts_high * (T - 1) - (T - 1)
    else:
        # k_high 阶 vs k_low 阶
        num_contexts_high = len(high_counts)
        num_contexts_low = len(low_counts)
        df = num_contexts_high * (T - 1) - num_contexts_low * (T - 1)
    
    # 确保自由度为正
    if df <= 0:
        return G2, None, df
    
    # p 值
    p_value = chi2.sf(G2, df)
    
    return G2, p_value, df


# ============================================================================
# Permutation Test（置换检验）- 增强版
# ============================================================================

def permutation_test_markov(tool_sequences, k_high, k_low=0, n_permutations=1000, random_seed=42):
    """
    通过置换检验验证 k_high 阶相对于 k_low 阶的改进
    零假设：k_high 阶和 k_low 阶无显著差异
    """
    np.random.seed(random_seed)
    
    # 计算观测数据的熵减
    entropy_results = analyze_entropy_reduction(tool_sequences)
    
    if k_high == 1 and k_low == 0:
        observed_delta = entropy_results['delta_H1']
        metric_name = 'delta_H1'
    elif k_high == 2 and k_low == 0:
        observed_delta = entropy_results['delta_H2_vs_0']
        metric_name = 'delta_H2_vs_0'
    elif k_high == 2 and k_low == 1:
        observed_delta = entropy_results['delta_H2_vs_1']
        metric_name = 'delta_H2_vs_1'
    else:
        raise ValueError(f"Unsupported comparison: {k_high}-order vs {k_low}-order")
    
    print(f"  Running {n_permutations} permutations for {k_high}-order vs {k_low}-order test...")
    
    # 生成零假设分布
    null_deltas = []
    
    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{n_permutations}")
        
        # 对每个序列内部的工具进行随机置换
        permuted_sequences = []
        for seq in tool_sequences:
            permuted = list(seq)
            np.random.shuffle(permuted)
            permuted_sequences.append(permuted)
        
        # 计算置换后的熵减
        perm_results = analyze_entropy_reduction(permuted_sequences)
        null_deltas.append(perm_results[metric_name])
    
    # 计算 p 值（单侧检验：观测值大于零假设）
    null_deltas = np.array(null_deltas)
    p_value = np.mean(null_deltas >= observed_delta)
    
    return observed_delta, null_deltas, p_value


# ============================================================================
# 时序交叉验证（新增）
# ============================================================================

def temporal_cross_validation(tool_sequences, k, n_folds=5):
    """
    时序阻塞交叉验证
    :param tool_sequences: 工具序列列表
    :param k: 马尔可夫阶数
    :param n_folds: 折数
    :return: 各折的对数似然和平均值
    """
    print(f"\n  Running {n_folds}-fold temporal cross-validation for {k}-order model...")
    
    # 合并所有序列为一个长序列（保持时序）
    full_sequence = []
    for seq in tool_sequences:
        full_sequence.extend(seq)
    
    n = len(full_sequence)
    fold_size = n // n_folds
    
    log_likelihoods = []
    
    for fold in range(n_folds):
        # 测试集：第 fold 折
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        
        # 训练集：除测试集外的所有数据
        train_seq = full_sequence[:test_start] + full_sequence[test_end:]
        test_seq = full_sequence[test_start:test_end]
        
        # 在训练集上估计转移概率
        if k == 0:
            # 0 阶：边际分布
            train_counts = Counter(train_seq)
            train_total = len(train_seq)
            train_probs = {tool: count / train_total for tool, count in train_counts.items()}
        else:
            # k 阶：转移概率
            train_transition_counts = defaultdict(lambda: defaultdict(int))
            train_context_totals = defaultdict(int)
            
            for i in range(len(train_seq) - k):
                if k == 1:
                    context = train_seq[i]
                else:
                    context = tuple(train_seq[i:i+k])
                next_tool = train_seq[i+k]
                train_transition_counts[context][next_tool] += 1
                train_context_totals[context] += 1
            
            train_probs = {}
            for context, next_tools in train_transition_counts.items():
                train_probs[context] = {
                    tool: count / train_context_totals[context]
                    for tool, count in next_tools.items()
                }
        
        # 在测试集上计算对数似然
        log_L = 0
        if k == 0:
            for tool in test_seq:
                p = train_probs.get(tool, 1e-10)  # 平滑
                log_L += np.log(p)
        else:
            for i in range(len(test_seq) - k):
                if k == 1:
                    context = test_seq[i]
                else:
                    context = tuple(test_seq[i:i+k])
                next_tool = test_seq[i+k]
                
                if context in train_probs and next_tool in train_probs[context]:
                    p = train_probs[context][next_tool]
                else:
                    p = 1e-10  # 平滑未见过的转移
                
                log_L += np.log(p)
        
        log_likelihoods.append(log_L)
        print(f"    Fold {fold+1}/{n_folds}: log-likelihood = {log_L:.2f}")
    
    avg_log_L = np.mean(log_likelihoods)
    print(f"  Average log-likelihood: {avg_log_L:.2f}")
    
    return log_likelihoods, avg_log_L


# ============================================================================
# AIC/BIC 计算
# ============================================================================

def compute_AIC_BIC(tool_sequences, k):
    """计算 k 阶模型的 AIC 和 BIC"""
    if k == 0:
        # 0 阶模型：边际分布
        tool_counts = Counter()
        n = 0
        for seq in tool_sequences:
            for tool in seq:
                tool_counts[tool] += 1
                n += 1
        
        log_L = 0
        for count in tool_counts.values():
            p = count / n
            log_L += count * np.log(p)
        
        T = len(tool_counts)
        num_params = T - 1
    else:
        # k 阶模型
        k_counts = defaultdict(lambda: defaultdict(int))
        n = 0
        
        for seq in tool_sequences:
            for i in range(len(seq) - k):
                if k == 1:
                    context = seq[i]
                else:
                    context = tuple(seq[i:i+k])
                next_tool = seq[i+k]
                k_counts[context][next_tool] += 1
                n += 1
        
        # 对数似然
        log_L = 0
        for context, next_tools in k_counts.items():
            context_total = sum(next_tools.values())
            for count in next_tools.values():
                if count > 0:
                    p = count / context_total
                    log_L += count * np.log(p)
        
        # 参数数量：实际观察到的上下文数 × (T-1)
        T = len(set(tool for seq in tool_sequences for tool in seq))
        num_params = len(k_counts) * (T - 1)
    
    # AIC 和 BIC
    AIC = -2 * log_L + 2 * num_params
    BIC = -2 * log_L + num_params * np.log(n)
    
    return AIC, BIC, num_params


# ============================================================================
# 可视化（新增）
# ============================================================================

def plot_entropy_reduction(entropy_results, output_dir='.'):
    """绘制熵减阶梯图"""
    k = [0, 1, 2]
    entropy = [entropy_results['H0'], entropy_results['H1'], entropy_results['H2']]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k, entropy, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Markov Order k', fontsize=12)
    plt.ylabel('Conditional Entropy (bits)', fontsize=12)
    plt.title('Entropy Reduction by Markov Order', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k)
    
    # 添加数值标注
    for i, (k_val, h_val) in enumerate(zip(k, entropy)):
        plt.text(k_val, h_val + 0.1, f'{h_val:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'entropy_reduction.png')
    plt.savefig(output_path, dpi=300)
    print(f"\n  Entropy reduction plot saved to: {output_path}")
    plt.close()


def plot_permutation_distribution(observed_delta, null_deltas, k_high, k_low, output_dir='.'):
    """绘制置换检验的零假设分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(null_deltas, bins=50, alpha=0.7, color='gray', edgecolor='black', label='Null distribution')
    plt.axvline(observed_delta, color='red', linestyle='--', linewidth=2, label=f'Observed ΔH = {observed_delta:.3f}')
    plt.axvline(np.mean(null_deltas), color='blue', linestyle=':', linewidth=2, label=f'Null mean = {np.mean(null_deltas):.3f}')
    
    plt.xlabel('Entropy Reduction (bits)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Permutation Test: {k_high}-order vs {k_low}-order', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'permutation_test_{k_high}vs{k_low}.png')
    plt.savefig(output_path, dpi=300)
    print(f"  Permutation test plot saved to: {output_path}")
    plt.close()


# ============================================================================
# 综合分析主函数（重构版）
# ============================================================================

def comprehensive_markov_analysis(tool_sequences, run_permutation=True, n_permutations=1000, 
                                   run_cv=True, output_dir='.'):
    """完整的马尔可夫依赖性分析（修正版）"""
    
    print("=" * 80)
    print("COMPREHENSIVE MARKOV DEPENDENCY ANALYSIS (CORRECTED)")
    print("=" * 80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {len(tool_sequences)}")
    print(f"  Total tool steps: {sum(len(seq) for seq in tool_sequences)}")
    print(f"  Average sequence length: {sum(len(seq) for seq in tool_sequences) / len(tool_sequences):.2f}")
    print(f"  Unique tools: {len(set(tool for seq in tool_sequences for tool in seq))}")
    
    # 1. 熵减分析
    print("\n" + "=" * 80)
    print("1. ENTROPY REDUCTION ANALYSIS")
    print("=" * 80)
    entropy_results = analyze_entropy_reduction(tool_sequences)
    
    print(f"\nEntropy values:")
    print(f"  H0 (0-order): {entropy_results['H0']:.4f} bits")
    print(f"  H1 (1-order): {entropy_results['H1']:.4f} bits")
    print(f"  H2 (2-order): {entropy_results['H2']:.4f} bits")
    
    print(f"\nEntropy reduction:")
    print(f"  ΔH1 (H0 - H1): {entropy_results['delta_H1']:.4f} bits ({entropy_results['relative_red_1']:.2f}% reduction)")
    print(f"  ΔH2 (H1 - H2): {entropy_results['delta_H2_vs_1']:.4f} bits ({entropy_results['relative_red_2_vs_1']:.2f}% additional reduction)")
    print(f"  ΔH2 (H0 - H2): {entropy_results['delta_H2_vs_0']:.4f} bits ({entropy_results['relative_red_2_vs_0']:.2f}% cumulative reduction) ⭐ NEW")
    
    # 绘图
    plot_entropy_reduction(entropy_results, output_dir)
    
    # 2. 似然比检验（修正版）
    print("\n" + "=" * 80)
    print("2. LIKELIHOOD RATIO TESTS (CORRECTED)")
    print("=" * 80)
    
    print("\n1st-order vs 0-order:")
    G2_1, p_lrt_1, df_1 = likelihood_ratio_test(tool_sequences, k_high=1, k_low=0)
    if G2_1 is not None and p_lrt_1 is not None:
        print(f"  G² statistic: {G2_1:.2f}")
        print(f"  Degrees of freedom: {df_1}")
        print(f"  P-value: {p_lrt_1:.4e}")
        print(f"  Significant: {'✅ Yes' if p_lrt_1 < 0.001 else '❌ No'} (α=0.001)")
    else:
        print(f"  G² statistic: {G2_1:.2f}")
        print(f"  ⚠️  df={df_1} may be invalid, rely on permutation test")
    
    print("\n2nd-order vs 1st-order:")
    G2_2_vs_1, p_lrt_2_vs_1, df_2_vs_1 = likelihood_ratio_test(tool_sequences, k_high=2, k_low=1)
    if G2_2_vs_1 is not None and p_lrt_2_vs_1 is not None:
        print(f"  G² statistic: {G2_2_vs_1:.2f}")
        print(f"  Degrees of freedom: {df_2_vs_1}")
        print(f"  P-value: {p_lrt_2_vs_1:.4e}")
        print(f"  Significant: {'✅ Yes' if p_lrt_2_vs_1 < 0.05 else '❌ No'} (α=0.05)")
    else:
        print(f"  G² statistic: {G2_2_vs_1:.2f}")
        print(f"  ⚠️  df={df_2_vs_1} may be invalid, rely on permutation test")
    
    print("\n2nd-order vs 0-order: ⭐ NEW")
    G2_2_vs_0, p_lrt_2_vs_0, df_2_vs_0 = likelihood_ratio_test(tool_sequences, k_high=2, k_low=0)
    if G2_2_vs_0 is not None and p_lrt_2_vs_0 is not None:
        print(f"  G² statistic: {G2_2_vs_0:.2f}")
        print(f"  Degrees of freedom: {df_2_vs_0}")
        print(f"  P-value: {p_lrt_2_vs_0:.4e}")
        print(f"  Significant: {'✅ Yes' if p_lrt_2_vs_0 < 0.001 else '❌ No'} (α=0.001)")
    else:
        print(f"  G² statistic: {G2_2_vs_0:.2f}")
        print(f"  ⚠️  df={df_2_vs_0} may be invalid, rely on permutation test")
    
    # 3. Permutation Test
    if run_permutation:
        print("\n" + "=" * 80)
        print("3. PERMUTATION TESTS (non-parametric, MOST RELIABLE)")
        print("=" * 80)
        
        print("\n1st-order vs 0-order:")
        obs_d1, null_dist_1, p_perm_1 = permutation_test_markov(
            tool_sequences, k_high=1, k_low=0, n_permutations=n_permutations
        )
        print(f"  Observed ΔH1: {obs_d1:.4f} bits")
        print(f"  Null distribution: {np.mean(null_dist_1):.4f} ± {np.std(null_dist_1):.4f}")
        print(f"  P-value: {p_perm_1:.4f}")
        print(f"  Significant: {'✅ Yes' if p_perm_1 < 0.001 else '❌ No'} (α=0.001)")
        plot_permutation_distribution(obs_d1, null_dist_1, 1, 0, output_dir)
        
        print("\n2nd-order vs 1st-order:")
        obs_d2_vs_1, null_dist_2_vs_1, p_perm_2_vs_1 = permutation_test_markov(
            tool_sequences, k_high=2, k_low=1, n_permutations=n_permutations
        )
        print(f"  Observed ΔH2: {obs_d2_vs_1:.4f} bits")
        print(f"  Null distribution: {np.mean(null_dist_2_vs_1):.4f} ± {np.std(null_dist_2_vs_1):.4f}")
        print(f"  P-value: {p_perm_2_vs_1:.4f}")
        print(f"  Significant: {'✅ Yes' if p_perm_2_vs_1 < 0.05 else '❌ No'} (α=0.05)")
        plot_permutation_distribution(obs_d2_vs_1, null_dist_2_vs_1, 2, 1, output_dir)
        
        print("\n2nd-order vs 0-order: ⭐ NEW")
        obs_d2_vs_0, null_dist_2_vs_0, p_perm_2_vs_0 = permutation_test_markov(
            tool_sequences, k_high=2, k_low=0, n_permutations=n_permutations
        )
        print(f"  Observed ΔH2: {obs_d2_vs_0:.4f} bits")
        print(f"  Null distribution: {np.mean(null_dist_2_vs_0):.4f} ± {np.std(null_dist_2_vs_0):.4f}")
        print(f"  P-value: {p_perm_2_vs_0:.4f}")
        print(f"  Significant: {'✅ Yes' if p_perm_2_vs_0 < 0.001 else '❌ No'} (α=0.001)")
        plot_permutation_distribution(obs_d2_vs_0, null_dist_2_vs_0, 2, 0, output_dir)
    
    # 4. 时序交叉验证（新增）
    if run_cv:
        print("\n" + "=" * 80)
        print("4. TEMPORAL CROSS-VALIDATION (NEW)")
        print("=" * 80)
        
        _, avg_log_L_0 = temporal_cross_validation(tool_sequences, k=0, n_folds=5)
        _, avg_log_L_1 = temporal_cross_validation(tool_sequences, k=1, n_folds=5)
        _, avg_log_L_2 = temporal_cross_validation(tool_sequences, k=2, n_folds=5)
        
        print(f"\nComparison:")
        print(f"  0-order avg log-likelihood: {avg_log_L_0:.2f}")
        print(f"  1-order avg log-likelihood: {avg_log_L_1:.2f} (Δ = {avg_log_L_1 - avg_log_L_0:.2f})")
        print(f"  2-order avg log-likelihood: {avg_log_L_2:.2f} (Δ = {avg_log_L_2 - avg_log_L_1:.2f})")
        print(f"  Best model by CV: {['0-order', '1-order', '2-order'][np.argmax([avg_log_L_0, avg_log_L_1, avg_log_L_2])]}")
    
    # 5. AIC/BIC
    print("\n" + "=" * 80)
    print("5. MODEL COMPARISON (AIC/BIC)")
    print("=" * 80)
    
    AIC_0, BIC_0, params_0 = compute_AIC_BIC(tool_sequences, k=0)
    AIC_1, BIC_1, params_1 = compute_AIC_BIC(tool_sequences, k=1)
    AIC_2, BIC_2, params_2 = compute_AIC_BIC(tool_sequences, k=2)
    
    print(f"\n0-order model:")
    print(f"  Parameters: {params_0}")
    print(f"  AIC: {AIC_0:.0f}")
    print(f"  BIC: {BIC_0:.0f}")
    
    print(f"\n1-order model:")
    print(f"  Parameters: {params_1}")
    print(f"  AIC: {AIC_1:.0f} (ΔAIC = {AIC_1-AIC_0:.0f})")
    print(f"  BIC: {BIC_1:.0f} (ΔBIC = {BIC_1-BIC_0:.0f})")
    print(f"  Preferred by AIC: {'✅ Yes' if AIC_1 < AIC_0 else '❌ No'}")
    print(f"  Preferred by BIC: {'✅ Yes' if BIC_1 < BIC_0 else '❌ No'}")
    
    print(f"\n2-order model:")
    print(f"  Parameters: {params_2}")
    print(f"  AIC: {AIC_2:.0f} (ΔAIC vs 0-order = {AIC_2-AIC_0:.0f}, vs 1-order = {AIC_2-AIC_1:.0f})")
    print(f"  BIC: {BIC_2:.0f} (ΔBIC vs 0-order = {BIC_2-BIC_0:.0f}, vs 1-order = {BIC_2-BIC_1:.0f})")
    print(f"  Preferred by AIC over 0-order: {'✅ Yes' if AIC_2 < AIC_0 else '❌ No'}")
    print(f"  Preferred by AIC over 1-order: {'✅ Yes' if AIC_2 < AIC_1 else '❌ No'}")
    print(f"  Preferred by BIC over 0-order: {'✅ Yes' if BIC_2 < BIC_0 else '❌ No'}")
    
    # 6. 结论
    print("\n" + "=" * 80)
    print("6. CONCLUSIONS")
    print("=" * 80)
    
    conclusions = []
    
    # 1阶依赖
    if entropy_results['delta_H1'] > 0.5 and (not run_permutation or p_perm_1 < 0.001):
        conclusions.append("✅ STRONG evidence for 1st-order Markov dependency (ΔH > 0.5 bits, p < 0.001)")
    
    # 2阶相对1阶
    if entropy_results['delta_H2_vs_1'] > 0.3 and (not run_permutation or p_perm_2_vs_1 < 0.05):
        conclusions.append("✅ Evidence for 2nd-order dependency BEYOND 1st-order (ΔH > 0.3 bits, p < 0.05)")
    
    # 2阶相对0阶（新增）
    if entropy_results['delta_H2_vs_0'] > 1.0 and (not run_permutation or p_perm_2_vs_0 < 0.001):
        conclusions.append("✅ STRONG evidence for 2nd-order dependency OVER 0-order (ΔH > 1.0 bits, p < 0.001) ⭐ NEW")
    
    # AIC判断
    if AIC_1 < AIC_0:
        conclusions.append("✅ 1-order model preferred by AIC over 0-order")
    
    if AIC_2 < AIC_1:
        conclusions.append("✅ 2-order model preferred by AIC over 1-order")
    elif AIC_2 > AIC_1:
        conclusions.append("⚠️  2-order model NOT preferred by AIC over 1-order (potential overfitting)")
    
    if AIC_2 < AIC_0:
        conclusions.append("✅ 2-order model preferred by AIC over 0-order ⭐ NEW")
    
    # 交叉验证
    if run_cv:
        best_k_cv = np.argmax([avg_log_L_0, avg_log_L_1, avg_log_L_2])
        conclusions.append(f"✅ Cross-validation prefers {['0-order', '1-order', '2-order'][best_k_cv]} model")
    
    print("\n📊 Summary:")
    for conclusion in conclusions:
        print(f"  {conclusion}")
    
    print("\n" + "=" * 80)
    
    return entropy_results


# ============================================================================
# 主程序
# ============================================================================

def main(trajectory_dir, filter_tools=None, remove_duplicates=False, 
         run_permutation=True, n_permutations=1000, run_cv=True, output_dir='.'):
    """
    主分析函数
    :param trajectory_dir: 轨迹文件目录
    :param filter_tools: 要过滤的工具列表
    :param remove_duplicates: 是否去除连续重复（建议 False）
    :param run_permutation: 是否运行置换检验（耗时）
    :param n_permutations: 置换检验次数
    :param run_cv: 是否运行时序交叉验证
    :param output_dir: 图表输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据
    all_sequences = load_all_trajectories(trajectory_dir)
    
    if not all_sequences:
        print("No sequences found. Exiting.")
        return
    
    # 提取工具序列
    print(f"\nExtracting tool sequences...")
    if filter_tools:
        print(f"  Filtering out tools: {filter_tools}")
    if remove_duplicates:
        print(f"  ⚠️  WARNING: Removing consecutive duplicates (may inflate entropy reduction)")
    else:
        print(f"  ✅ Keeping consecutive duplicates (recommended)")
    
    tool_sequences = extract_tool_sequences(
        all_sequences, 
        filter_tools=filter_tools,
        remove_consecutive_duplicates=remove_duplicates
    )
    
    # 运行综合分析
    comprehensive_markov_analysis(
        tool_sequences, 
        run_permutation=run_permutation,
        n_permutations=n_permutations,
        run_cv=run_cv,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # 配置参数
    DEFAULT_TRAJECTORY_DIR = '/home/jjy/AutoTool/AgentBoard/agentboard/examples/visualisation/trajectories'
    OUTPUT_DIR = './markov_analysis_results'
    
    # 运行分析
    main(
        trajectory_dir=DEFAULT_TRAJECTORY_DIR,
        filter_tools=['unknown'],  # 过滤 'unknown' 工具
        remove_duplicates=False,   # ✅ 保留连续重复（真实行为）
        run_permutation=True,      # 运行置换检验（最可靠）
        n_permutations=1000,       # 置换次数
        run_cv=True,               # 运行时序交叉验证
        output_dir=OUTPUT_DIR      # 输出目录
    )