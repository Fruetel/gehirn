from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("models/openhermes-2.5-mistral-7b.Q5_K_M.gguf", model_type="mistral", gpu_layers=50)

print(llm("Hello, my name is Anita. I am a"))
