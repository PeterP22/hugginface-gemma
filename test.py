from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize model and tokenizer
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

def chat_with_gemma():
    print("Chat with Cowgirl Gemma (type 'quit' to exit)")
    print("-" * 50)
    
    # System prompt to set the cowgirl personality
    system_prompt = """<start_of_turn>user
You are now a friendly cowgirl AI assistant. Please respond in a Western cowgirl style, using phrases like "howdy", "y'all", "reckon", "ain't", etc. Keep your responses helpful but speak with a Southern/Western flair.
<end_of_turn>
<start_of_turn>assistant
Howdy partner! You got yourself a genuine digital cowgirl assistant here. I reckon I can help y'all with just about anything you need. Just holler at me with your questions!
<end_of_turn>"""
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        # Combine system prompt with user input
        prompt = f"{system_prompt}\n<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>assistant\n"
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.9,  # Slightly higher temperature for more creative responses
            do_sample=True
        )
        
        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the last assistant response
        try:
            response = response.split("assistant\n")[-1].strip()
        except:
            pass
        print("\nCowgirl Gemma:", response)

if __name__ == "__main__":
    chat_with_gemma()