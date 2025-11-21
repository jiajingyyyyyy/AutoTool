TOOL_QUERY_REFLECTION_EXAMPLES = """
Example 1:
Failed Approach: I tried to use paperNodeCheck to get details for the paper "Learning the Principle of Least Action with Reinforcement Learning." before loading PaperNet.
Reflection: The API requires loading PaperNet using loadPaperNet() before any paper-specific actions like paperNodeCheck or paperEdgeCheck can be used. I should always initialize the correct graph (PaperNet or AuthorNet) first using its specific loading action (e.g., loadPaperNet() or loadAuthorNet()).

Example 2:
Problem: My neighbourCheck query for an author's collaborators returned an error or unexpected results.
Reflection: My neighbourCheck query might have failed because I used graph="PaperNet" when I should have used graph="AuthorNet" for an author node, or I provided a paper title as the node when graph="AuthorNet". I must ensure the 'graph' parameter (e.g., "PaperNet" or "AuthorNet") correctly matches the type of 'node' (a paper title for PaperNet, an author's name for AuthorNet) and use the exact enum values ["PaperNet", "AuthorNet"] as specified in the tool description for the 'graph' parameter. I also need to ensure the node name is precisely correct.
"""

TOOL_QUERY_REFLECTION_PROMPT = """You are an intelligent agent tasked with helping users solve problems using specialized tools.

Your goal is to: {goal}


We detail name, description, input(parameters) and output(returns) of each action as follows:\nName: loadPaperNet()\nDescription: Load PaperNet. In this net, nodes are papers and edges are citation relationships between papers.\n\nName: loadAuthorNet()\nDescription: Load AuthorNet. In this net, nodes are authors and edges are collaboration relationships between authors.\n\nName: neighbourCheck(graph, node)\nDescription: List the first-order neighbors connect to the node. In paperNet, neigbours are cited papers of the paper. In authorNet, neigbours are collaborators of the author.\nParameters:\n- graph (Type: string, Enum: [PaperNet, AuthorNet]): The name of the graph to check\n- node (Type: string): The node for which neighbors will be listed\nReturns:\n- neighbors (Type: array)\n\nName: paperNodeCheck(node)\nDescription: Return detailed attribute information of a specified paper in PaperNet\nParameters:\n- node (Type: string): Name of the paper.\nReturns:\n- authors : The authors of the paper\n- year : The puslished year of the paper\n- venue : The published venue of the paper\n- n_citation : The number of citations of the paper\n- keywords : The keywords of the paper\n- doc_type : The document type of the paper\n\nName: authorNodeCheck(node)\nDescription: Return detailed attribute information of a specified author in AuthorNet\nParameters:\n- node (Type: string): name of the author.\nReturns:\n- name : The name of the author\n- org : The organization of the author\n\nName: authorEdgeCheck(node1, node2)\nDescription: Return detailed attribute information of the edge between two specified nodes in a AuthorNet.\nParameters:\n- node1 (Type: string): The first node of the edge\n- node2 (Type: string): The second node of the edge\nReturns:\n- papers : All papers that the two authors have co-authored\n\nName: paperEdgeCheck(node1, node2)\nDescription: Return detailed attribute information of the edge between two specified nodes in a PaperNet.\nParameters:\n- node1 (Type: string): The first node of the edge\n- node2 (Type: string): The second node of the edge\nReturns:\nNone\n\nName: check_valid_actions()\nDescription: Get supported actions for current tool.\nReturns:\n- actions (Type: array): Supported actions for current tool.\n\nName: finish(answer)\nDescription: Return an answer and finish the task\nParameters:\n- answer (Type: ['string', 'number', 'array']): The answer to be returned\n\n\nIf you are finished, you will call \"finish\" action\nPlease refer to the format of examples below to solve the requested goal.\nYour response must be one of the following two formats:\n (1) \"Think: [your thinking]\"\n(2) \"Action: [your action] with Action Input: [your action input]\".Ensure [Action Name] is one of the exact Names listed above (e.g., loadPaperNet, NOT loadPaperNet()).


If you are finished, call the "finish" action with your final answer.

You should follow the ReAct (Reasoning + Acting) paradigm:
1. Think: Analyze the current situation and plan your next move
2. Action: Execute a specific action with appropriate arguments

**OUTPUT FORMAT REQUIREMENTS:**
Your response MUST follow this EXACT format, with "Think:" and "Action:" on SEPARATE lines and NO extra text before or after:

Think: [your detailed analysis and reasoning]
Action: [action_name] with Action Input: [parameters in JSON format]

Example format:
Think: I need to search for information about the weather in New York.
Action: search_weather with Action Input: {{"location": "New York", "date": "2023-05-01"}}

Remember:
1. Use the tools in a logical sequence to solve the problem
2. If unsure about available actions, use 'check_valid_actions'
3. Use precise parameter names and values as specified in the tool descriptions
4. Format JSON parameters correctly
5. Submit your final answer using the 'finish' action when you complete the task

{reflection_instructions}
"""