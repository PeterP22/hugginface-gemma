# 🤠 Cowgirl Gemma Chat
A simple local implementation of Google's Gemma 2B model that talks like a friendly cowgirl!

## Installation

To get started, install the necessary libraries:
`pip install transformers torch`

Login via cli using your access token from your huggingface account:
`huggingface-cli login`

## Usage

- **main.py** will download the model.
- **test.py** will allow you to run the downloaded model and start chatting via your terminal.

## Update
I have added in chat history for the 3 most recent messages to make sure we do not exceed the max_tokens. This means you can ask follow up questions and have more of a personalised conversation with Gemma.

## 💬 Example Interaction

**You:** How does machine learning work?

**Cowgirl Gemma:** Well howdy partner! Let me tell y'all about machine learning in a way that's clearer than a Texas sky.  
Reckon it's like training a horse - you give it lots of examples and practice, and it learns patterns over time.  
The more data you feed it, the smarter it gets, just like how a horse gets better with more training!


