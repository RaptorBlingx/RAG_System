from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(query, documents):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    input_text = " ".join([doc['content'] for doc in documents]) + " " + query
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs, max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    query = "What is Creamobile?"
    
    # Assume final_results is obtained from previous step
    final_results = [...]  # Replace with actual results
    
    response = generate_response(query, final_results)
    print(response)

if __name__ == "__main__":
    main()
