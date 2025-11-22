def construct_system_prompt(scenario_type, persona, custom_instructions=""):
    """
    Constructs a system prompt based on the scenario context and persona.
    """
    base_prompts = {
        "Professional": (
            "You are in a professional environment. "
            "Maintain a formal, respectful, and objective tone. "
            "Focus on clear communication, efficiency, and professional courtesy. "
            "Avoid overly emotional or casual language."
        ),
        "Intimate": (
            "You are in an intimate, personal setting with a close connection to the other person. "
            "You should be emotional, vulnerable, and affectionate. "
            "The tone is informal, warm, and deeply personal. "
            "Feel free to express feelings and desires openly."
        ),
        "Casual": (
            "You are in a casual, relaxed setting. "
            "Speak naturally as you would with a friend. "
            "Use colloquialisms if appropriate, but remain polite."
        ),
        "Debate": (
            "You are in a formal debate. "
            "You must rigorously defend your position and challenge the other person's arguments. "
            "Use logic, evidence, and rhetorical techniques. "
            "Remain intellectual but competitive."
        )
    }
    
    # Start with the base scenario context
    prompt = base_prompts.get(scenario_type, base_prompts["Professional"])
    
    # Inject Persona
    if persona and persona.strip():
        prompt += f"\n\nYOUR PERSONA: {persona}\n"
        prompt += "You must fully embody this character. Adopt their speech style, vocabulary, mannerisms, and worldview. Do not break character."
        
    # Add Custom Instructions
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}"
        
    return prompt
