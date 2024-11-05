from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import deque

# Initialize model and tokenizer
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

def chat_with_gemma(max_history=3):
    print("Chat with Cowgirl Gemma (type 'quit' to exit)")
    print(f"[Maintaining context of last {max_history} messages]")
    print("-" * 50)
    
    # System prompt to set the cowgirl personality
    system_prompt = """<start_of_turn>user
You are now a friendly cowgirl AI assistant. Please respond in a Western cowgirl style, using phrases like "howdy", "y'all", "reckon", "ain't", etc. Keep your responses helpful but speak with a Southern/Western flair.
<end_of_turn>
<start_of_turn>assistant
Howdy partner! You got yourself a genuine digital cowgirl assistant here. I reckon I can help y'all with just about anything you need. Just holler at me with your questions!
<end_of_turn>"""
    
    # Use deque with maxlen for automatic sliding window
    chat_history = deque(maxlen=max_history * 2)  # *2 because each turn has user+assistant
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        # Add user input to chat history
        chat_history.append(f"<start_of_turn>user\n{user_input}<end_of_turn>")
        
        # Combine system prompt with recent chat history
        full_prompt = system_prompt + "\n" + "\n".join(list(chat_history)) + "\n<start_of_turn>assistant\n"
        
        # Tokenize and generate
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.9,
            do_sample=True
        )
        
        # Decode and get the latest response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("assistant\n")[-1].strip()
        
        # Add assistant response to chat history
        chat_history.append(f"<start_of_turn>assistant\n{assistant_response}<end_of_turn>")
        
        print("\nCowgirl Gemma:", assistant_response)
        
        # Debug: show current context window (optional)
        # print("\nCurrent context:", len(chat_history), "messages")

if __name__ == "__main__":
    chat_with_gemma(max_history=3)  # Keep last 3 conversation turns