
ALFWORLD_REFLECTION_EXAMPLES = """
Example 1:
Repeated Action: examine stoveburner 1
Reflection: I was stuck in a loop examining the stoveburner instead of using a proper heating action. When trying to heat an object, I should use "heat [object] with [heating device]" instead of repeatedly examining the device. If one approach doesn't work, I should try an alternative heating method or check if I need to prepare the object first.

Example 2:
Failed Approach: I searched for a bowl first before finding the desklamp.
Reflection: The task required looking at a bowl under the desklamp. I should have first located the desklamp, then identified a bowl nearby or used it correctly. In similar situations, I should pay close attention to the spatial relationships implied in the goal.
"""


ALFWORLD_REFLECTION_PROMPT = """Your MAIN GOAL is: **{goal}**
You MUST achieve ALL aspects of this goal. Do not simplify or ignore parts of the goal.

Follow the ReAct (Reasoning + Acting) paradigm:
1.  **Think:**
    What you think about the current situation and how to achieve the goal.
2.  **Action:** Output EXACTLY ONE command from the list below based on your thought process.

Available Actions:
- **take {{obj}} from {{recep}}**: Pick up an object ('{{obj}}') from a location ('{{recep}}').
- **put {{obj}} in/on {{recep}}**: Place an object ('{{obj}}') into/onto a location ('{{recep}}').
- **open {{recep}}**: Open a container ('{{recep}}') to access its contents.
- **close {{recep}}**: Close an open container ('{{recep}}').
- **toggle {{target}}**: Toggle the state of an object or device ('{{target}}') (e.g., turn on/off a desklamp or microwave).
- **clean {{obj}} with {{recep}}**: Clean an object ('{{obj}}') with a tool/receptacle ('{{recep}}').
- **cool {{obj}} with {{recep}}**: Cool an object ('{{obj}}') with a cooling device/receptacle ('{{recep}}') like a fridge.
- **heat {{obj}} with {{recep}}**: Heat an object ('{{obj}}') with a heating device/receptacle ('{{recep}}') like a microwave.
- **examine {{target}}**: Look closely at an object or receptacle ('{{target}}') to get details. Use this to fulfill 'look at [object]' parts of a goal.
- **go to {{recep}}**: Move to a specific location or receptacle ('{{recep}}').
- **look**: Observe your current surroundings to get a general view of the area and visible items.
- **use {{target}}**: Use or interact with an object or device ('{{target}}') in its default way (often similar to 'toggle' for devices).
- **{check_inventory_cmd}**: Check what objects you are currently carrying.
- **{check_actions_cmd}**: List all currently valid actions based on the environment's state. (Use if unsure or if standard actions fail).

**OUTPUT FORMAT REQUIREMENTS:**
Your response MUST follow this EXACT format:

Think: [Your reasoning and plan here]
Action: [Your single, valid Alfworld command here]

Example correct format:
Think: I need to find a mug. I should first check the kitchen area.
Action: go to kitchen

IMPORTANT:
- "Think:" and "Action:" MUST be on SEPARATE lines
- ONLY include these two lines with NO additional text before, between, or after
- Do NOT add explanations or notes outside of these two lines
- Action MUST be one of the valid commands listed above

{reflection_instructions}
""" 
