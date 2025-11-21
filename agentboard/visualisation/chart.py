import json
from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

# 尝试导入 ToolGraph
try:
    from autool.core.tool_predict.datastruct import ToolGraph
except ImportError:
    print("错误: 无法从 datastruct.py 导入 ToolGraph。")
    print("请确保 datastruct.py 在同一目录或在 PYTHONPATH 中。")
    exit()


# ============================================================================
# 饼图绘制函数（使用你提供的版本）
# ============================================================================

def plot_successor_pie_chart(entity_name: any,
                             successors_map: dict,
                             total_frequency: int,
                             output_dir: str,
                             entity_type: str = "pair"):
    """
    生成最终的出版质量饼图。
    - 图例阴影线为白色，带黑色边框
    - 所有 >= 10% 的切片都有独特的阴影图案
    - 颜色和阴影规则一致应用
    """
    # --- 1. 样式设置 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['hatch.linewidth'] = 1.5
    # 核心修改: 设置全局默认阴影颜色为白色
    plt.rcParams['hatch.color'] = 'white'

    # --- 2. 数据准备 ---
    filtered_successors_map = {}
    if entity_type == "pair":
        tool_B = entity_name[1]
        filtered_successors_map = {succ: freq for succ, freq in successors_map.items() if succ != tool_B}
    elif entity_type == "single":
        tool_A = entity_name
        filtered_successors_map = {succ: freq for succ, freq in successors_map.items() if succ != tool_A}
    
    total_frequency = sum(filtered_successors_map.values())
    if total_frequency == 0:
        print(f"    跳过 {entity_name}: 过滤后没有后继数据")
        return

    sorted_successors = sorted(filtered_successors_map.items(), key=lambda item: item[1], reverse=True)
    labels = [succ.replace("_", " ") for succ, freq in sorted_successors]
    sizes = [freq for succ, freq in sorted_successors]
    percentages = [(s / total_frequency) * 100 if total_frequency > 0 else 0 for s in sizes]
    if not sizes:
        return

    # --- 3. 动态颜色分配 ---
    light_orange_color = '#FDB562'
    blue_color = 'tab:blue'
    green_color = 'tab:green'
    special_colors = ['tab:orange', 'tab:blue', 'tab:green']
    other_colors_pool = [c for c in plt.get_cmap('tab10').colors if c not in special_colors]
    
    final_colors = []
    for i in range(len(sizes)):
        if i == 0:
            final_colors.append(light_orange_color)
        elif i == 1:
            final_colors.append(blue_color)
        elif i == 2:
            final_colors.append(green_color)
        else:
            final_colors.append(other_colors_pool[(i - 3) % len(other_colors_pool)])

    # --- 4. 动态阴影分配（所有 >= 10% 的切片）---
    hatches_ordered = ['xx', '/', '\\', 'o', 'O', '.', '*', '+', '|']
    assigned_hatches = []
    hatch_idx = 0
    for i in range(len(sizes)):
        if percentages[i] >= 10:
            assigned_hatches.append(hatches_ordered[hatch_idx % len(hatches_ordered)])
            hatch_idx += 1
        else:
            assigned_hatches.append('')  # 小切片无阴影

    # --- 5. 创建图形并绘制 ---
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(top=0.8)

    def autopct_conditional(pct):
        return f'{pct:.1f}%' if pct >= 10 else ''

    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct=autopct_conditional,
        startangle=90,
        pctdistance=0.85,
        explode=[0.02] * len(sizes),
        colors=final_colors
    )
    
    # --- 6. 样式循环 ---
    for i, wedge in enumerate(wedges):
        wedge.set_hatch(assigned_hatches[i])
        wedge.set_edgecolor('white')
        wedge.set_linewidth(1.5)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(32)
        autotext.set_fontweight('bold')

    ax.axis('equal')

    # --- 7. 居中标题和过滤图例 ---
    if entity_type == "pair":
        title_str = f"{entity_name[0].replace('_', ' ')} → {entity_name[1].replace('_', ' ')} ({total_frequency})"
    else:
        title_str = f"{entity_name.replace('_', ' ')} ({total_frequency})"
    fig.suptitle(title_str, fontsize=40, fontweight='bold', y=0.97)
    
    legend_elements = []
    for i, label in enumerate(labels):
        if percentages[i] >= 10:
            legend_patch = Patch(
                facecolor=final_colors[i],
                edgecolor='black', 
                hatch=assigned_hatches[i],
                label=label
            )
            legend_elements.append(legend_patch)

    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=min(len(legend_elements), 5),
        frameon=True,
        edgecolor='black',
        fontsize=22
    )
    
    # --- 8. 保存图形 ---
    if entity_type == "pair":
        filename_prefix = f"successors_of_{entity_name[0]}_then_{entity_name[1]}"
    else:
        filename_prefix = f"successors_of_{entity_name}"
    
    filename = f"{filename_prefix}_final.png".replace(" ", "_").replace("→", "to")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"    饼图已保存到: {filepath}")
    plt.close(fig)

    # 重置 rcParams
    plt.rcdefaults()


# ============================================================================
# 参数惯性表生成函数
# ============================================================================

def generate_parameter_inertia_table(tool_graph, target_tool_name: str):
    """
    生成并打印给定目标工具的输入参数来源表
    """
    if target_tool_name not in tool_graph.nodes:
        print(f"  错误: 目标工具 '{target_tool_name}' 在工具图中未找到。无法生成参数惯性表。")
        return

    target_node = tool_graph.nodes[target_tool_name]
    target_input_params = list(target_node.input_params.keys()) 

    if not target_input_params:
        print(f"  工具 '{target_tool_name}' 没有定义的输入参数。")
        return

    print(f"  工具参数惯性: {target_tool_name}")
    print("  " + "-" * 90)
    print(f"  {'目标参数':<20} | {'来源工具':<20} | {'来源参数(输出)':<25} | {'频率':<10} | {'占比':<10}")
    print("  " + "-" * 90)

    found_any_dependency = False

    if target_tool_name not in tool_graph.param_edges:
        print(f"  工具 '{target_tool_name}' 没有记录的参数依赖边。")
        for target_param_name in target_input_params:
            print(f"  {target_param_name:<20} | {'N/A':<20} | {'N/A':<25} | {'0':<10} | {'0.0%':<10}")
        print("  " + "-" * 90)
        return

    param_dependencies_for_target_tool = tool_graph.param_edges[target_tool_name]

    for target_param_name in target_input_params:
        if target_param_name in param_dependencies_for_target_tool:
            sources_for_this_param = param_dependencies_for_target_tool[target_param_name]
            
            if not sources_for_this_param:
                print(f"  {target_param_name:<20} | {'无具体来源':<20} | {'N/A':<25} | {'-':<10} | {'-':<10}")
                continue

            total_frequency_for_this_param = sum(edge.count for edge in sources_for_this_param.values())
            
            if total_frequency_for_this_param == 0: 
                print(f"  {target_param_name:<20} | {'找到来源但频率为0':<20} | {'N/A':<25} | {'-':<10} | {'-':<10}")
                continue

            sorted_sources = sorted(sources_for_this_param.items(), key=lambda item: item[1].count, reverse=True)
            
            first_source = True
            for (source_tool, source_param_name), param_edge_obj in sorted_sources:
                found_any_dependency = True
                proportion = (param_edge_obj.count / total_frequency_for_this_param) * 100 if total_frequency_for_this_param > 0 else 0
                
                param_display_name = target_param_name if first_source else "" 
                print(f"  {param_display_name:<20} | {source_tool:<20} | {source_param_name:<25} | {param_edge_obj.count:<10} | {proportion:>9.1f}%")
                first_source = False
            
            if not first_source: 
                if len(sources_for_this_param) > 1:
                    print(f"  {'':<20} | {'-'*20} | {'-'*25} | {'-'*10} | {'-'*10}")

        else: 
            print(f"  {target_param_name:<20} | {'无记录来源':<20} | {'N/A':<25} | {'0':<10} | {'0.0%':<10}")

    print("  " + "-" * 90)


# ============================================================================
# 主函数
# ============================================================================

def main(tool_description_path: str, 
         tool_trajectory_path: str, 
         high_freq_tool_edge_threshold: int = 5,
         tool_pairs_for_pie_chart: list = None,
         single_tools_for_pie_chart: list = None,
         tool_for_parameter_inertia_table: str = None,
         output_pie_charts_dir: str = "tool_successor_pie_charts"):
    """
    主分析函数
    :param tool_description_path: 工具描述 JSON 文件路径
    :param tool_trajectory_path: 轨迹文件目录路径
    :param high_freq_tool_edge_threshold: 高频工具边的阈值
    :param tool_pairs_for_pie_chart: 要生成饼图的工具对列表 [(tool_A, tool_B), ...]
    :param single_tools_for_pie_chart: 要生成饼图的单个工具列表 [tool_A, tool_B, ...]
    :param tool_for_parameter_inertia_table: 要生成参数惯性表的工具名称
    :param output_pie_charts_dir: 饼图输出目录
    """
    # 1. 初始化 ToolGraph
    tool_graph = ToolGraph()
    tool_graph.debug = False

    # 2. 加载工具描述
    print(f"--- 从以下路径加载工具描述: {tool_description_path} ---")
    if not os.path.exists(tool_description_path):
        print(f"错误: 工具描述文件未找到: {tool_description_path}")
        return
    tool_graph.load_tool_description_from_json(tool_description_path)
    if not tool_graph.nodes:
        print("错误: 未加载工具描述。请检查文件格式和内容。退出。")
        return
    print(f"成功加载 {len(tool_graph.nodes)} 个工具描述。")

    # 3. 加载工具轨迹并更新图
    file_list = [os.path.abspath(os.path.join(tool_trajectory_path, f)) 
                 for f in os.listdir(tool_trajectory_path) if f.endswith('.json')]

    print(f"\n找到 {len(file_list)} 个轨迹文件")

    for trajectory_file in file_list:
        print(f"\n--- 从以下路径加载工具轨迹: {trajectory_file} ---")
        if not os.path.exists(trajectory_file):
            print(f"错误: 轨迹文件未找到: {trajectory_file}")
            continue
            
        try:
            with open(trajectory_file, "r", encoding="utf-8") as f:
                trajectory_data = json.load(f)
            
            sequences_list = trajectory_data.get("sequences")
            
            if isinstance(sequences_list, list):
                print(f"找到 {len(sequences_list)} 个序列待处理。")
                if not sequences_list:
                    print(f"警告: '{trajectory_file}' 中的 'sequences' 列表为空")
                for i, seq_data_item in enumerate(sequences_list):
                    if not isinstance(seq_data_item, dict):
                        print(f"警告: 序列项 {i} 不是字典，跳过。")
                        continue
                    tool_graph.update_graph(seq_data_item)
            elif isinstance(trajectory_data, dict) and "steps" in trajectory_data:
                print("将轨迹文件作为单个序列处理...")
                tool_graph.update_graph(trajectory_data)
            else:
                print(f"警告: 无法在 '{trajectory_file}' 中找到 'sequences' 键下的序列列表，也不是单个序列字典。")
                if isinstance(trajectory_data, list):
                    for i, seq_data_item in enumerate(trajectory_data):
                        if not isinstance(seq_data_item, dict):
                            print(f"警告: 序列项 {i} 不是字典，跳过。")
                            continue
                        tool_graph.update_graph(seq_data_item)

        except Exception as e:
            print(f"加载或处理轨迹文件 {trajectory_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 4. 分析所有工具对的后继（A -> B -> [后继]）
    print(f"\n--- 分析所有工具对的后继（A -> B -> [后继]）---")
    tool_pair_successors_freq = defaultdict(lambda: defaultdict(int))

    if not tool_graph.paths:
        print("  未记录工具路径，无法分析工具对后继。")
    else:
        for path_obj in tool_graph.paths:
            tools_in_path = path_obj.tools
            path_frequency = path_obj.frequency
            if len(tools_in_path) >= 3:
                for i in range(len(tools_in_path) - 2):
                    tool_A = tools_in_path[i]
                    tool_B = tools_in_path[i+1]
                    tool_C = tools_in_path[i+2]
                    
                    tool_pair = (tool_A, tool_B)
                    tool_pair_successors_freq[tool_pair][tool_C] += path_frequency

    # 5. 分析单个工具的后继（A -> [后继]）
    print(f"\n--- 分析单个工具的后继（A -> [后继]）---")
    single_tool_successors_freq = defaultdict(lambda: defaultdict(int))

    if not tool_graph.paths:
        print("  未记录工具路径，无法分析单个工具后继。")
    else:
        for path_obj in tool_graph.paths:
            tools_in_path = path_obj.tools
            path_frequency = path_obj.frequency
            if len(tools_in_path) >= 2:
                for i in range(len(tools_in_path) - 1):
                    tool_A = tools_in_path[i]
                    tool_B = tools_in_path[i+1]
                    single_tool_successors_freq[tool_A][tool_B] += path_frequency

    # 6. 生成工具对的饼图
    if tool_pairs_for_pie_chart:
        print(f"\n--- 为工具对生成后继饼图 ---")
        if not os.path.exists(output_pie_charts_dir):
            os.makedirs(output_pie_charts_dir)
            print(f"  创建目录: {output_pie_charts_dir}")

        for pair_to_plot in tool_pairs_for_pie_chart:
            tool_A, tool_B = pair_to_plot
            if pair_to_plot in tool_pair_successors_freq:
                successors_map = tool_pair_successors_freq[pair_to_plot]
                total_freq_for_pair = sum(successors_map.values())
                if total_freq_for_pair > 0:
                    print(f"  为以下工具对生成饼图: {tool_A} -> {tool_B}")
                    plot_successor_pie_chart(
                        entity_name=pair_to_plot,
                        successors_map=successors_map,
                        total_frequency=total_freq_for_pair,
                        output_dir=output_pie_charts_dir,
                        entity_type="pair"
                    )
                else:
                    print(f"  跳过 {tool_A} -> {tool_B}: 未找到后继或总频率为零。")
            else:
                print(f"  跳过 {tool_A} -> {tool_B}: 在分析的工具对后继中未找到该对。")

    # 7. 生成单个工具的饼图
    if single_tools_for_pie_chart:
        print(f"\n--- 为单个工具生成后继饼图 ---")
        if not os.path.exists(output_pie_charts_dir):
            os.makedirs(output_pie_charts_dir)
            print(f"  创建目录: {output_pie_charts_dir}")

        for tool_to_plot in single_tools_for_pie_chart:
            if tool_to_plot in single_tool_successors_freq:
                successors_map = single_tool_successors_freq[tool_to_plot]
                total_freq_for_tool = sum(successors_map.values())
                if total_freq_for_tool > 0:
                    print(f"  为工具生成饼图: {tool_to_plot}")
                    plot_successor_pie_chart(
                        entity_name=tool_to_plot,
                        successors_map=successors_map,
                        total_frequency=total_freq_for_tool,
                        output_dir=output_pie_charts_dir,
                        entity_type="single"
                    )
                else:
                    print(f"  跳过 {tool_to_plot}: 未找到后继或总频率为零。")
            else:
                print(f"  跳过 {tool_to_plot}: 在分析的单个工具后继中未找到该工具。")

    # 8. 生成参数惯性表
    if tool_for_parameter_inertia_table:
        print(f"\n--- 为工具生成参数惯性表: '{tool_for_parameter_inertia_table}' ---")
        generate_parameter_inertia_table(tool_graph, tool_for_parameter_inertia_table)


if __name__ == "__main__":
    # --- 配置 ---
    DEFAULT_TOOL_DESC_FILE = '/home/jjy/AutoTool/AgentBoard/FastToolCalling/src/AutoTool/graph/tool_predict/tool_doc/scienceworld_tool_description.json'
    DEFAULT_TRAJECTORY_FILE = '/home/jjy/AutoTool/AgentBoard/agentboard/examples/visualisation/trajectories'

    # 使用环境变量或直接路径
    tool_desc_file = os.getenv('TOOL_DESC_PATH', DEFAULT_TOOL_DESC_FILE)
    trajectory_file = os.getenv('TRAJECTORY_PATH', DEFAULT_TRAJECTORY_FILE)
    
    # 检查默认文件是否存在
    if not os.path.exists(tool_desc_file):
        print(f"警告: 默认工具描述文件未找到: {tool_desc_file}")
        tool_desc_file = input(f"请输入工具描述 JSON 文件的路径: ")
        if not os.path.exists(tool_desc_file):
            print(f"错误: 工具描述文件未找到: '{tool_desc_file}'。退出。")
            exit()

    if not os.path.exists(trajectory_file):
        print(f"警告: 默认轨迹文件/目录未找到: {trajectory_file}")
        trajectory_file = input(f"请输入轨迹 JSON 文件目录的路径: ")
        if not os.path.exists(trajectory_file):
            print(f"错误: 轨迹文件/目录未找到: '{trajectory_file}'。退出。")
            exit()
            
    frequency_threshold_for_tool_edges = 3
    
    # 配置可视化选项
    tool_pairs_for_pie_chart = [
        ("focus_on", "wait"), 
        ("go_to", "look_around"), 
    ]
    
    single_tools_for_pie_chart = [
        "go_to",
        "open",
    ]
    
    tool_for_parameter_inertia_table = "use"
    output_pie_charts_dir = "tool_successor_pie_charts"
    
    # 运行主函数
    main(
        tool_description_path=tool_desc_file,
        tool_trajectory_path=trajectory_file,
        high_freq_tool_edge_threshold=frequency_threshold_for_tool_edges,
        tool_pairs_for_pie_chart=tool_pairs_for_pie_chart,
        single_tools_for_pie_chart=single_tools_for_pie_chart,
        tool_for_parameter_inertia_table=tool_for_parameter_inertia_table,
        output_pie_charts_dir=output_pie_charts_dir
    )