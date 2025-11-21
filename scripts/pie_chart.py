import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch


def plot_successor_pie_chart(entity_name: any,
                             successors_map: dict,
                             total_frequency: int,
                             output_dir: str,
                             entity_type: str = "pair"
                             ):
    """
    Generates a final, publication-quality pie chart.
    - Legend hatches are white with a black border.
    - All slices >= 10% receive a distinct hatch pattern.
    - Color and hatch rules are applied consistently.
    """
    # --- 1. Style Setup ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['hatch.linewidth'] = 1.5
    # --- 核心修改: Set the GLOBAL default hatch color to white ---
    plt.rcParams['hatch.color'] = 'white'

    # --- 2. Data Preparation ---
    # ... (Filtering logic remains the same) ...
    filtered_successors_map = {}
    if entity_type == "pair":
        tool_B = entity_name[1]
        filtered_successors_map = {succ: freq for succ, freq in successors_map.items() if succ != tool_B}
    elif entity_type == "single":
        tool_A = entity_name
        filtered_successors_map = {succ: freq for succ, freq in successors_map.items() if succ != tool_A}
    
    total_frequency = sum(filtered_successors_map.values())
    if total_frequency == 0: return

    sorted_successors = sorted(filtered_successors_map.items(), key=lambda item: item[1], reverse=True)
    labels = [succ.replace("_", " ") for succ, freq in sorted_successors]
    sizes = [freq for succ, freq in sorted_successors]
    percentages = [(s / total_frequency) * 100 if total_frequency > 0 else 0 for s in sizes]
    if not sizes: return

    # --- 3. Dynamic Color Assignment ---
    # ... (Color assignment logic remains the same) ...
    light_orange_color = '#FDB562'
    blue_color = 'tab:blue'
    green_color = 'tab:green'
    special_colors = ['tab:orange', 'tab:blue', 'tab:green']
    other_colors_pool = [c for c in plt.get_cmap('tab10').colors if c not in special_colors]
    
    final_colors = []
    for i in range(len(sizes)):
        if i == 0: final_colors.append(light_orange_color)
        elif i == 1: final_colors.append(blue_color)
        elif i == 2: final_colors.append(green_color)
        else: final_colors.append(other_colors_pool[(i - 3) % len(other_colors_pool)])

    # --- 4. 核心修改: Dynamic Hatch Assignment for ALL Slices >= 10% ---
    hatches_ordered = ['xx', '/', '\\', 'o', 'O', '.', '*', '+', '|']
    assigned_hatches = []
    hatch_idx = 0
    for i in range(len(sizes)):
        if percentages[i] >= 10:
            assigned_hatches.append(hatches_ordered[hatch_idx % len(hatches_ordered)])
            hatch_idx += 1
        else:
            assigned_hatches.append('') # No hatch for small slices

    # --- 5. Create Figure and Plot ---
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
    
    # --- 6. Styling Loop ---
    for i, wedge in enumerate(wedges):
        # Apply the dynamically assigned hatch
        wedge.set_hatch(assigned_hatches[i])
        wedge.set_edgecolor('white')
        wedge.set_linewidth(1.5)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(32)
        autotext.set_fontweight('bold')

    ax.axis('equal')

    # --- 7. Centered Title and Filtered Legend ---
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
                # 核心修改: edgecolor is for the border, hatch color is now globally white
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
    
    # --- 8. Save Figure ---
    if entity_type == "pair":
        filename_prefix = f"successors_of_{entity_name[0]}_then_{entity_name[1]}"
    else:
        filename_prefix = f"successors_of_{entity_name}"
    
    filename = f"{filename_prefix}_final_v3.png".replace(" ", "_").replace("→", "to")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"    Final styled pie chart (v3) saved to: {filepath}")
    plt.close(fig)

    # Reset rcParams if you have other plots in the same script
    plt.rcdefaults()
