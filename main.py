from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("models/openhermes-2.5-mistral-7b.Q5_K_M.gguf", model_type="mistral", gpu_layers=50, context_length=8192)

prompt = """<|im_start|>system
You are a specialized AI language model. Your job is to identify logical flaws
in a given text. IMPORTANT: Your response MUST be in JSON format, even
intermediate steps.
You will use train of thought to do this. First, summarize the
text in your own words. Then, identify any logical flaws in the text. Finally,
write a short paragraph explaining how the text could be improved.
Your response has to be in the folowing format:

{
    "summary": "This is a summary of the text.",
    "flaws": [
        "This is a flaw in the text.",
        "This is another flaw in the text."
    ],
    "improvement": "This is how the text could be improved."
}

NEVER deviate from this response format. Do not add any additional
information.<|im_end|>
<|im_start|>user
Please tell me how the following text could be improved:
In the quaint village of Logica, it is a well-known fact that all villagers either always tell the truth or always lie. The village elder claims, "I am a liar." If we take a moment to analyze this statement, it presents a paradox. Should we believe the elder? If we do, then according to his own admission, he must be lying, which would mean he’s actually a truth-teller. But that can't be, because then he wouldn't lie about being a liar! 🤯 The statement folds in on itself and flutters like a confused butterfly, unable to decide on which flower to land. Logic seems to twist like the village's winding paths, leading us to an impossible conclusion: that the elder is both a liar and a truth-teller.

Now, consider this: every Tuesday, the villagers of Logica play a game where they can only say statements that are the opposite of what they would normally say. On one such Tuesday, the village gossip, who is a known truth-teller, declares, "I will lie next Wednesday." This presents a tantalizing tangle. If the gossip is telling the truth, then she will indeed lie the next day, which is consistent with the rules of the game. However, this would mean that she is lying about lying the next day, which implies that she would actually tell the truth, contradicting the premise of the Tuesday game. It’s a brain-teasing loop, where truth and deception dance in a masquerade, with neither willing to unmask first.<|im_end>
<|im_start|>assistant
"""

print(llm(prompt))
