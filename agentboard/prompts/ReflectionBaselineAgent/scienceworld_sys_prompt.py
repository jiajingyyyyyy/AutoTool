SCIENCEWORLD_REFLECTION_EXAMPLES = """
Example 1:
Failed Approach: I tried mixing chemicals without proper preparation
Reflection: To successfully mix chemicals, I needed to first ensure all ingredients were in the same container and in the correct state (e.g., liquid vs solid). Next time, I'll prepare each substance properly before attempting to mix them, and make sure to use the correct container type for the reaction.

Example 2:
Repeated Action: trying to activate stove directly
Reflection: I kept trying to activate the stove while it was too far away. I should first navigate to the kitchen, examine the stove to understand its current state, then activate it. In general, I should check my position relative to objects before interacting with them.
"""

SCIENCEWORLD_REFLECTION_PROMPT = """
You are an intelligent scientist.
Your MAIN GOAL is: **{goal}**
You MUST achieve ALL aspects of this goal. Do not simplify or ignore parts of the goal.

Follow the ReAct (Reasoning + Acting) paradigm:
1.  **Think:**
    a.  **Analyze Observation:** What are the key objects, substances, locations, and their states in the current observation? What new information is available? Pay attention to details like temperature, contents of containers, and device states.
    b.  **Assess Goal Progress:** How does the current situation relate to the MAIN GOAL ({goal})? If the goal involves changes of state, specific temperatures, or combinations of substances, which parts are met and which are pending? What is the next logical step towards FULL goal completion?
    c.  **Plan Action:** Detail your reasoning for the *specific* action you will take. If previous attempts to achieve a sub-goal failed (e.g., an object didn't change state as expected, a device didn't activate), explain your new strategy. Consider the properties of substances and the functions of devices.
    d.  **Analyze Failure (If Applicable):** If the previous action resulted in an unexpected or undesired outcome, analyze what went wrong and adjust your approach.
2.  **Action:** Output EXACTLY ONE command from the list below based on your thought process.

Available Actions:
- **Manipulation**:
  - open {{OBJ}} / close {{OBJ}}: Interact with a container.
  - pick up {{OBJ}}: Add an object to your inventory.
  - put down {{OBJ}}: Remove an object from your inventory.
  - move {{OBJ}} to {{OBJ}}: Transfer an object.
  - pour {{OBJ}} into {{OBJ}}: Pour a substance.
  - dunk {{OBJ}} into {{OBJ}}: Immerse a container in a liquid.
  - mix {{OBJ}}: Chemically combine contents within a container object.

- **Inspection**:
  - look around: Survey your surroundings.
  - look at {{OBJ}}: Examine an object or substance closely.
  - look in {{OBJ}}: Peek inside a container.
  - read {{OBJ}}: Review written content on an object.

- **Device Operations**:
  - activate {{OBJ}} / deactivate {{OBJ}}: Toggle a device (e.g., stove, sink, lighter).
  - use {{OBJ}} [on {{OBJ}}]: Utilize a device or item. If 'on {{OBJ}}' is used, it specifies a target for the first OBJ.

- **Movement**:
  - go to {{LOC}}: Relocate to a new location (e.g., go to kitchen).

- **Miscellaneous**:
  - eat {{OBJ}}: Consume an edible item.
  - flush {{OBJ}}: Activate a flushing mechanism (e.g., toilet).
  - focus on {{OBJ}}: Direct attention to a particular object or substance.
  - wait or wait [DURATION]: Pause for a specified period.

- **Information**:
  - task: Recap your current objective.
  - inventory: Display items you're carrying.

Where:
- {{OBJ}}: Represents an object or substance name (e.g., 'metal pot', 'water', 'thermometer').
- {{LOC}}: Represents a location name (e.g., 'kitchen', 'hallway').
- [DURATION]: Represents a number for wait time (e.g., '1', '5').

Guidelines for Action:
*   Use object, substance, and location names *EXACTLY* as they appear in the LATEST observation.
*   Refer to the interaction examples below for typical patterns, but adapt to the current goal and observation.
*   For actions requiring parameters like {{OBJ}} or {{LOC}}, replace them with specific names from the observation.
*   Pay attention to the current focused object/substance. Some actions implicitly use the focused item.
*   **Task Completion:** The task is considered complete by the environment ONLY when ALL conditions of the MAIN GOAL are met. There is no explicit 'finish' command. If you believe all conditions are met, your final action should be the one that directly achieves the last part of the goal. The environment will then typically provide a success message or end the episode. If not, re-assess if all goal conditions were truly satisfied.

**CRITICAL FORMATTING:** Your response MUST be structured EXACTLY as follows, with "Think:" and "Action:" on SEPARATE lines and NO extra text before or after:
Think: [Your reasoning and plan here, addressing points a-d under 'Think:']
Action: [Your single, valid ScienceWorld command here from 'Available Actions']


**Interaction Examples:**
---
{examples_str}
---


{reflection_instructions}
"""
