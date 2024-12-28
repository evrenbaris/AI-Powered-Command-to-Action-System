from transformers import pipeline

# Pretrained NLP model for text-to-action
nlp_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Example command
command = "Deploy a UAV to Region B for reconnaissance."

# Extract key information
questions = [
    {"question": "What is the task?", "context": command},
    {"question": "What is the target region?", "context": command},
    {"question": "What equipment is mentioned?", "context": command},
]

# Analyze the command
print("Command Analysis:")
for q in questions:
    answer = nlp_model(question=q["question"], context=q["context"])
    print(f"Question: {q['question']}")
    print(f"Answer: {answer['answer']}\n")
