import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np

# Load FAQ data from CSV
def load_faq_data(csv_file):
    df = pd.read_csv(csv_file)
    questions = df['question'].tolist()
    answers = dict(zip(df['question'], df['answer']))
    return questions, answers

# Load the FAQ data
questions, answers = load_faq_data('data.csv')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Get embeddings for the FAQ questions
faq_embeddings = get_embeddings(questions)

# Create a FAISS index
index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(faq_embeddings)

# Function to retrieve the most relevant FAQ
def retrieve_faq(query):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, 1)
    return questions[indices[0][0]], answers[questions[indices[0][0]]]

# Function to retrieve FAQ with context
def retrieve_faq_with_context(conversation_history, query):
    full_context = " ".join([entry['user'] for entry in conversation_history] + [query])
    return retrieve_faq(full_context)

# Chatbot function
def chatbot():
    print("Welcome to the NAVER Smart Store FAQ chatbot! How can I help you today?")
    conversation_history = []

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Check if the user input is appropriate
        if not user_input or any(word in user_input.lower() for word in ["bye", "hello", "hi"]):
            print("Chatbot: I'm a chatbot for Smart Store FAQs, please ask me a question about Smart Store.")
            continue

        # Retrieve the most relevant FAQ answer with context
        retrieved_question, retrieved_answer = retrieve_faq_with_context(conversation_history, user_input)

        # Save to conversation history
        conversation_history.append({"user": user_input, "bot": retrieved_answer})

        # Display the bot's response
        print(f"chatbot: {retrieved_answer}")

if __name__ == "__main__":
    chatbot()
