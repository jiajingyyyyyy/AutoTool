import json
from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt

# Attempt to import ToolGraph from datastruct. Assume datastruct.py is in the same directory or accessible via PYTHONPATH
try:
    from autool.core.tool_predict.datastruct import ToolGraph
except ImportError:
    print("Error: Could not import ToolGraph from datastruct.py.")
    print("Please ensure datastruct.py is in the same directory as this script, or in your PYTHONPATH.")
    exit()

# Helper function to generate and save pie chart
def plot_successor_pie_chart(pair: tuple, successors_map: dict, total_frequency: int, output_dir: str):
    """
    Generates and saves a pie chart for the successors of a given tool pair.
    """
    tool_A, tool_B = pair
    labels = []
    sizes = []
    other_count = 0
    threshold_percent = 2.0 # Group small slices into 'Other'

    # Sort successors by frequency to make 'Other' consistent if used
    sorted_successors = sorted(successors_map.items(), key=lambda item: item[1], reverse=True)

    for successor, freq in sorted_successors:
        percentage = (freq / total_frequency) * 100
        if percentage < threshold_percent and len(labels) > 5 : # Avoid 'Other' if there are few items or it's a large slice
            other_count += freq
        else:
            labels.append(f"{successor}\\n({freq}, {percentage:.1f}%)")
            sizes.append(freq)
    
    if other_count > 0:
        labels.append(f"Other\\n({other_count}, {(other_count / total_frequency) * 100:.1f}%)")
        sizes.append(other_count)

    if not sizes: # No successors to plot
        print(f"    No data to plot for {tool_A} -> {tool_B}")
        return

    fig, ax = plt.subplots(figsize=(10, 7)) # Adjusted for better label readability
    
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, pctdistance=0.85) 

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f"Successor Distribution for: {tool_A} -> {tool_B}\\nTotal Occurrences of Pair leading to Successor: {total_frequency}", pad=20)
    
    ax.legend(wedges, labels, title="Successors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout(rect=[0, 0, 0.75, 1]) 

    filename = f"successors_of_{tool_A}_then_{tool_B}.png".replace(" ", "_").replace(":", "_")
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath)
        print(f"    Pie chart saved to: {filepath}")
    except Exception as e:
        print(f"    Error saving pie chart {filepath}: {e}")
    plt.close(fig)


# Helper function to generate and print parameter inertia table
def generate_parameter_inertia_table(tool_graph, target_tool_name: str):
    """
    Generates and prints a table showing the sources of input parameters for a given target tool.
    """
    if target_tool_name not in tool_graph.nodes:
        print(f"  Error: Target tool '{target_tool_name}' not found in the tool graph. Cannot generate parameter inertia table.")
        return

    target_node = tool_graph.nodes[target_tool_name]
    target_input_params = list(target_node.input_params.keys()) 

    if not target_input_params:
        print(f"  Tool '{target_tool_name}' has no defined input parameters.")
        return

    print(f"  Parameter Inertia for Tool: {target_tool_name}")
    print("  ------------------------------------------------------------------------------------")
    print(f"  {'Target Param':<20} | {'Source Tool':<20} | {'Source Param (Output)':<25} | {'Frequency':<10} | {'Proportion':<10}")
    print("  ------------------------------------------------------------------------------------")

    found_any_dependency = False

    if target_tool_name not in tool_graph.param_edges:
        print(f"  No parameter dependency edges recorded for tool '{target_tool_name}'.")
        for target_param_name in target_input_params:
            print(f"  {target_param_name:<20} | {'N/A':<20} | {'N/A':<25} | {'0':<10} | {'0.0%':<10}")
        print("  ------------------------------------------------------------------------------------")
        return

    param_dependencies_for_target_tool = tool_graph.param_edges[target_tool_name]

    for target_param_name in target_input_params:
        if target_param_name in param_dependencies_for_target_tool:
            sources_for_this_param = param_dependencies_for_target_tool[target_param_name]
            
            if not sources_for_this_param:
                print(f"  {target_param_name:<20} | {'No specific sources':<20} | {'N/A':<25} | {'-':<10} | {'-':<10}")
                continue

            total_frequency_for_this_param = sum(edge.count for edge in sources_for_this_param.values())
            
            if total_frequency_for_this_param == 0: 
                 print(f"  {target_param_name:<20} | {'Sources found but total freq is 0':<20} | {'N/A':<25} | {'-':<10} | {'-':<10}")
                 continue

            sorted_sources = sorted(sources_for_this_param.items(), key=lambda item: item[1].count, reverse=True)
            
            first_source = True
            # print(sorted_sources)
            for (source_tool, source_param_name), param_edge_obj in sorted_sources:
                found_any_dependency = True
                proportion = (param_edge_obj.count / total_frequency_for_this_param) * 100 if total_frequency_for_this_param > 0 else 0
                
                param_display_name = target_param_name if first_source else "" 
                print(f"  {param_display_name:<20} | {source_tool:<20} | {source_param_name:<25} | {param_edge_obj.count:<10} | {proportion:>9.1f}%")
                first_source = False
            
            if not first_source : 
                if len(sources_for_this_param) >1 : print(f"  {'':<20} | {'-'*20} | {'-'*25} | {'-'*10} | {'-'*10}")

        else: 
            print(f"  {target_param_name:<20} | {'No recorded sources':<20} | {'N/A':<25} | {'0':<10} | {'0.0%':<10}")
    
    if not found_any_dependency and any(target_param_name not in param_dependencies_for_target_tool for target_param_name in target_input_params):
        pass 

    print("  ------------------------------------------------------------------------------------") 

def main(tool_description_path: str, 
         tool_trajectory_path: str, 
         high_freq_tool_edge_threshold: int = 5,
         tool_pairs_for_pie_chart: list = None,
         tool_for_parameter_inertia_table: str = None,
         output_pie_charts_dir: str = "tool_successor_pie_charts"
         ):
    # 1. 初始化 ToolGraph
    tool_graph = ToolGraph()
    tool_graph.debug = False # Set to True for more verbose output from ToolGraph methods

    # 2. 加载工具描述
    print(f"--- Loading Tool Descriptions from: {tool_description_path} ---")
    if not os.path.exists(tool_description_path):
        print(f"Error: Tool description file not found at {tool_description_path}")
        return
    tool_graph.load_tool_description_from_json(tool_description_path)
    if not tool_graph.nodes:
        print("Error: No tool descriptions loaded. Check the file format and content. Exiting.")
        return
    print(f"Successfully loaded {len(tool_graph.nodes)} tool descriptions.")

    file_list = [os.path.abspath(os.path.join(tool_trajectory_path, f)) for f in os.listdir(tool_trajectory_path)]

    for trajectory_file in file_list:
    # 3. 加载工具轨迹并更新图
        print(f"\n--- Loading Tool Trajectories from: {trajectory_file} ---")
        if not os.path.exists(trajectory_file):
            print(f"Error: Tool trajectory file not found at {trajectory_file}")
            # Continue without trajectories if you want to analyze just the tool descriptions
            # or handle this case as an error. For now, we'll return.
            return
            
        try:
            with open(trajectory_file, "r", encoding="utf-8") as f:
                trajectory_data = json.load(f)
            
            # The structure of trajectory_data can vary.
            # Based on datastruct.py, update_graph expects a single sequence dictionary.
            # If your file contains a list of sequences under a "sequences" key:
            sequences_list = trajectory_data.get("sequences")
            
            if isinstance(sequences_list, list):
                print(f"Found {len(sequences_list)} sequences to process.")
                if not sequences_list:
                    print(f"Warning: 'sequences' list is empty in {trajectory_file}")
                for i, seq_data_item in enumerate(sequences_list):
                    if not isinstance(seq_data_item, dict):
                        print(f"Warning: Sequence item {i} is not a dictionary, skipping.")
                        continue
                    # print(f"\nProcessing sequence {i+1}/{len(sequences_list)}...")
                    tool_graph.update_graph(seq_data_item)
            elif isinstance(trajectory_data, dict) and "steps" in trajectory_data: # If the file itself is a single sequence
                print("Processing the trajectory file as a single sequence...")
                tool_graph.update_graph(trajectory_data)
            else:
                print(f"Warning: Could not find a list of sequences under 'sequences' key, nor a single sequence dict in {tool_trajectory_path}. Check file structure.")
                print("Attempting to process file as a list of sequences if it's a list directly.")
                if isinstance(trajectory_data, list):
                    for i, seq_data_item in enumerate(trajectory_data):
                        if not isinstance(seq_data_item, dict):
                            print(f"Warning: Sequence item {i} is not a dictionary, skipping.")
                            continue
                        print(f"\nProcessing sequence {i+1}/{len(trajectory_data)}...")
                        tool_graph.update_graph(seq_data_item)


        except Exception as e:
            print(f"Error loading or processing trajectories from {trajectory_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 打印 ToolGraph 的 to_json() 输出 (可选，可能非常长)
    # print(f"\n--- ToolGraph JSON Representation (Summary) ---")
    # # For brevity, let's print counts instead of the full JSON if it's too large
    # print(f"  Number of tools (nodes): {len(tool_graph.nodes)}")
    # edge_count = sum(len(targets) for targets in tool_graph.edges.values())
    # print(f"  Number of tool call edges: {edge_count}")
    # print(f"  Number of unique tool paths recorded: {len(tool_graph.paths)}")
    # param_edge_count = 0
    # for target_tool_data in tool_graph.param_edges.values():
    #     for source_map in target_tool_data.values():
    #         param_edge_count += len(source_map)
    # print(f"  Number of unique parameter dependency edges: {param_edge_count}")
    # To print the full JSON (can be very large):
    # full_json_output = tool_graph.to_json()
    # with open("tool_graph_output.json", "w", encoding="utf-8") as f_out:
    #     f_out.write(full_json_output)
    # print("Full ToolGraph JSON representation saved to tool_graph_output.json")

    # Print N most frequent tool paths
    # print(f"\n--- Most Frequent Tool Paths ---")
    # num_top_paths_to_print = 10 # Or make this a parameter to main()
    # if not tool_graph.paths:
    #     print("  No tool paths recorded in the graph.")
    # else:
    #     # Sort paths by frequency in descending order
    #     sorted_paths = sorted(tool_graph.paths, key=lambda p: p.frequency, reverse=True)
    #     print(f"  Top {min(num_top_paths_to_print, len(sorted_paths))} most frequent paths (out of {len(sorted_paths)} unique paths):")
    #     for i, path_obj in enumerate(sorted_paths[:num_top_paths_to_print]):
    #         path_str = " -> ".join(path_obj.tools)
    #         print(f"    {i+1}. Path: [{path_str}], Frequency: {path_obj.frequency}")

    # # Analyze successors for a specific tool pair relationship
    # print(f"\n--- Successor Analysis for Specific Tool Pair ---")
    # # Define the tool pair: (source_tool_of_interest, then_analyze_successors_of_this_tool)
    # # For example, if we want to see if "go_to" -> "look_around" exists, 
    # # and if so, what follows "look_around".
    # source_tool_of_interest = "go_to"  # Change as needed
    # target_whose_successors_to_analyze = "look_around" # Change as needed

    # print(f"  Analyzing if '{source_tool_of_interest}' -> '{target_whose_successors_to_analyze}' exists, and then successors of '{target_whose_successors_to_analyze}'.")

    # # Check if source_tool_of_interest exists
    # if source_tool_of_interest not in tool_graph.nodes:
    #     print(f"    Source tool '{source_tool_of_interest}' not found in the graph.")
    # # Check if source_tool_of_interest has outgoing edges and if target_whose_successors_to_analyze is among them
    # elif source_tool_of_interest not in tool_graph.edges or \
    #      target_whose_successors_to_analyze not in tool_graph.edges[source_tool_of_interest]:
    #     print(f"    No direct call edge found from '{source_tool_of_interest}' to '{target_whose_successors_to_analyze}'.")
    # else:
    #     # The A -> B relationship exists, print its frequency
    #     call_count_A_to_B = tool_graph.edges[source_tool_of_interest][target_whose_successors_to_analyze].call_count
    #     print(f"    Confirmed: '{source_tool_of_interest}' -> '{target_whose_successors_to_analyze}' exists with {call_count_A_to_B} calls.")

    #     # Now, analyze successors of target_whose_successors_to_analyze (Tool B)
    #     print(f"\n    Analyzing successors of '{target_whose_successors_to_analyze}':")
    #     tool_B_name = target_whose_successors_to_analyze
    #     if tool_B_name not in tool_graph.nodes:
    #         # This case should be rare if the A->B edge exists, but good to check
    #         print(f"      Tool '{tool_B_name}' (the target of the pair) not found as a node in the graph, though an edge to it exists.")
    #     elif tool_B_name in tool_graph.edges and tool_graph.edges[tool_B_name]:
    #         print(f"      Direct successors of '{tool_B_name}' and their call frequencies:")
    #         successors_of_B = tool_graph.edges[tool_B_name]
    #         sorted_successors_of_B = sorted(successors_of_B.items(), key=lambda item: item[1].call_count, reverse=True)
    #         if not sorted_successors_of_B:
    #             print(f"        Tool '{tool_B_name}' has outgoing edges defined but no specific successors listed (empty target dict).") # Should not happen if edges[tool_B_name] is not empty
    #         for target_tool, edge_data in sorted_successors_of_B:
    #             print(f"        -> {target_tool}: {edge_data.call_count} calls")
    #     else:
    #         print(f"      Tool '{tool_B_name}' has no recorded successors in the graph.")

    # # Analyze successors for all 2-tool sequences (tool pairs)
    # print(f"\n--- Successor Analysis for All Tool Pairs (A -> B -> [Successors]) ---")
    # # Data structure to store: {(tool_A, tool_B): {tool_C: frequency_of_C_after_AB}}
    # tool_pair_successors_freq = defaultdict(lambda: defaultdict(int))

    # if not tool_graph.paths:
    #     print("  No tool paths recorded, cannot analyze tool pair successors.")
    # else:
    #     for path_obj in tool_graph.paths:
    #         tools_in_path = path_obj.tools
    #         path_frequency = path_obj.frequency
    #         if len(tools_in_path) >= 3:
    #             for i in range(len(tools_in_path) - 2):
    #                 tool_A = tools_in_path[i]
    #                 tool_B = tools_in_path[i+1]
    #                 tool_C = tools_in_path[i+2]
                    
    #                 tool_pair = (tool_A, tool_B)
    #                 tool_pair_successors_freq[tool_pair][tool_C] += path_frequency
        
    #     if not tool_pair_successors_freq:
    #         print("  No 3-tool sequences found in paths to analyze successors of pairs.")
    #     else:
    #         # Calculate total outgoing frequency for each (A, B) pair
    #         # Structure: [((A,B), total_outgoing_freq_from_AB, {C: freq, D: freq}), ...]
    #         pair_outgoing_analysis = []
    #         for pair, successors_map in tool_pair_successors_freq.items():
    #             total_outgoing_freq = sum(successors_map.values())
    #             pair_outgoing_analysis.append((pair, total_outgoing_freq, successors_map))
            
    #         # Sort pairs by their total outgoing frequency (i.e., how often the A->B sequence leads to *any* C)
    #         sorted_pairs_by_total_freq = sorted(pair_outgoing_analysis, key=lambda x: x[1], reverse=True)
            
    #         num_top_pairs_to_detail = 10 # How many top (A,B) pairs to detail their successors
    #         print(f"\n  Details for Top {min(num_top_pairs_to_detail, len(sorted_pairs_by_total_freq))} Most Frequent Tool Pairs (A -> B) and their Successors:")

    #         for i, (pair, total_freq, successors_map) in enumerate(sorted_pairs_by_total_freq[:num_top_pairs_to_detail]):
    #             print(f"\n    {i+1}. Pair: ({pair[0]} -> {pair[1]}) (This pair leads to a successor {total_freq} times in total)")
                
    #             # Sort successors of this specific pair by their frequency
    #             sorted_successors_for_this_pair = sorted(successors_map.items(), key=lambda item: item[1], reverse=True)
                
    #             if not sorted_successors_for_this_pair:
    #                 print("      No specific successors recorded for this pair (this should not happen if total_freq > 0).")
    #             else:
    #                 print("      Successors from this pair:")
    #                 for successor_tool, freq in sorted_successors_for_this_pair:
    #                     percentage = (freq / total_freq) * 100 if total_freq > 0 else 0
    #                     print(f"        -> {successor_tool}: {freq} times ({percentage:.1f}% of this pair's continuations)")

    # # 5. 识别并打印高频工具调用边
    # print(f"\n--- High-Frequency Tool Edges (Call Count >= {high_freq_tool_edge_threshold}) ---")
    # high_frequency_tool_edges = []
    # if not tool_graph.edges:
    #     print("  No tool call edges recorded in the graph.")
    # else:
    #     for source_tool, targets in tool_graph.edges.items():
    #         for target_tool, edge_data in targets.items():
    #             if edge_data.call_count >= high_freq_tool_edge_threshold:
    #                 print(f"  {source_tool} -> {target_tool}: {edge_data.call_count} calls")
    #                 high_frequency_tool_edges.append((source_tool, target_tool))
    #     if not high_frequency_tool_edges:
    #         print(f"  No tool edges found with call count >= {high_freq_tool_edge_threshold}.")

    # # --- Begin: New Section for Pie Chart Visualization ---
    # print(f"\n--- Generating Successor Pie Charts ---")
    # if not os.path.exists(output_pie_charts_dir):
    #     os.makedirs(output_pie_charts_dir)
    #     print(f"  Created directory: {output_pie_charts_dir}")

    # # tool_pair_successors_freq is calculated in the "Successor Analysis for All Tool Pairs" section
    # # Structure: tool_pair_successors_freq = defaultdict(lambda: defaultdict(int))
    # # {(tool_A, tool_B): {tool_C: frequency_of_C_after_AB}}

    # if not tool_pair_successors_freq and tool_pairs_for_pie_chart:
    #     print("  No tool pair successor data available (tool_pair_successors_freq is empty), cannot generate pie charts.")
    # else:
    #     for pair_to_plot in tool_pairs_for_pie_chart:
    #         tool_A, tool_B = pair_to_plot
    #         if pair_to_plot in tool_pair_successors_freq:
    #             successors_map = tool_pair_successors_freq[pair_to_plot]
    #             total_freq_for_pair = sum(successors_map.values())
    #             if total_freq_for_pair > 0:
    #                 print(f"  Generating pie chart for successors of: {tool_A} -> {tool_B}")
    #                 plot_successor_pie_chart(pair_to_plot, successors_map, total_freq_for_pair, output_pie_charts_dir)
    #             else:
    #                 print(f"  Skipping pie chart for {tool_A} -> {tool_B}: No successors found or zero total frequency.")
    #         else:
    #             print(f"  Skipping pie chart for {tool_A} -> {tool_B}: Pair not found in analyzed tool pair successors.")
    # # --- End: New Section for Pie Chart Visualization ---


    # --- Begin: New Section for Parameter Inertia Table ---
    print(f"\n--- Generating Parameter Inertia Table for Tool: '{tool_for_parameter_inertia_table}' ---")
    
    generate_parameter_inertia_table(tool_graph, tool_for_parameter_inertia_table)
    # --- End: New Section for Parameter Inertia Table ---



if __name__ == "__main__":
    # --- Configuration ---
    # Please replace these paths with the actual paths to your files.
    # Ensure datastruct.py is in the same directory or accessible in PYTHONPATH.

    # Example for AlfWorld (using paths from your datastruct.py example)
    DEFAULT_TOOL_DESC_FILE = '/home/jjy/AutoTool/AgentBoard/FastToolCalling/src/AutoTool/graph/tool_predict/tool_doc/scienceworld_tool_description.json'
    DEFAULT_TRAJECTORY_FILE = '/home/jjy/AutoTool/AgentBoard/agentboard/examples/visualisation/trajectories' # Example, may need a different file for multiple sequences or different structure

    # Use environment variables or direct paths
    tool_desc_file = os.getenv('TOOL_DESC_PATH', DEFAULT_TOOL_DESC_FILE)
    trajectory_file = os.getenv('TRAJECTORY_PATH', DEFAULT_TRAJECTORY_FILE)
    
    # Check if default files exist, otherwise prompt user (or use placeholder)
    if not os.path.exists(tool_desc_file):
        print(f"Warning: Default tool description file not found: {tool_desc_file}")
        tool_desc_file = input(f"Please enter the path to your tool description JSON file: ")
        if not os.path.exists(tool_desc_file):
            print(f"Error: Tool description file not found at '{tool_desc_file}'. Exiting.")
            exit()

    if not os.path.exists(trajectory_file):
        print(f"Warning: Default trajectory file not found: {trajectory_file}")
        trajectory_file = input(f"Please enter the path to your tool trajectory JSON file: ")
        if not os.path.exists(trajectory_file):
            print(f"Error: Trajectory file not found at '{trajectory_file}'. Exiting.")
            exit()
            
    frequency_threshold_for_tool_edges = 3 # Minimum call_count for a tool edge to be "high-frequency"
    
    # New configuration options for visualizations
    tool_pairs_for_pie_chart = [("go_to", "look_around"), ("open", "look_at"), ("pick_up", "inventory")] # Example pairs, ensure these are likely to exist
    tool_for_parameter_inertia_table = "use" # Example tool, ensure it has input params and is used
    output_pie_charts_dir = "tool_successor_pie_charts" # Directory to save pie charts

    # --- End Configuration ---

    main(tool_desc_file, trajectory_file, frequency_threshold_for_tool_edges, \
         tool_pairs_for_pie_chart, tool_for_parameter_inertia_table, output_pie_charts_dir) 