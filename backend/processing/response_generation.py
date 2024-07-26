from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import time

def remove_repetition(text):
    # Use regex to remove consecutive duplicate phrases
    text = re.sub(r'\b(\w+\s+){1,3}\b(?=\1)', '', text)
    # Remove exact duplicate sentences
    sentences = text.split('. ')
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    return '. '.join(unique_sentences)

def generate_response(query, documents):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Summarize the content of the documents
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarized_docs = []
    for doc in documents:
        summary = summarizer(doc['content'], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summarized_docs.append(summary)

    # Combine summarized text with the query
    input_text = " ".join(summarized_docs[:3]) + " " + query  # Limit to first 3 summaries to manage length
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove repetition
    response = remove_repetition(response)
    
    return response

def main():
    query = "What is Creamobile?"
    
    # Assume final_results is obtained from previous step
    final_results = [...]  # Replace with actual results
    
    response = generate_response(query, final_results)
    print(response)

if __name__ == "__main__":
    main()
