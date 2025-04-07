import ollama
import os
import json
import numpy as np
from numpy.linalg import norm

def parse_file(file_path):
    with open(file_path, encoding="utf-8-sig") as f:
        para=[]
        buffer=[]
        for line in f.readlines():
            line=line.strip()
            if line:
                buffer.append(line)
            elif len(buffer) > 0:
                para.append(" ".join(buffer))
                buffer=[]
        if len(buffer):
            para.append(" ".join(buffer))
        return para

def save_embeddings(embeddings, file_path):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # Extract the base name of the file path
    base_name = os.path.basename(file_path)
    with open(f"embeddings/{base_name}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(file_path):
    if not os.path.exists(f"embeddings/{file_path}.json"):
        return False
    with open(f"embeddings/{file_path}.json", "r") as f:
        return json.load(f)

def get_embeddings(file_path, modelname, chunks):
    if (embeddings := load_embeddings(file_path)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)['embedding']
    for chunk in chunks
    ]
    save_embeddings(embeddings, file_path)
    return embeddings

def find_similar(needle, haystack):
    needle_norm=norm(needle)
    similarities=[
        np.dot(needle, item)/(norm(needle)*norm(item)) for item in haystack
    ]
    return sorted(zip(similarities, range(len(similarities))), reverse=True)
def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:"""
    
    file_path = input("Enter the file path: ")
    para = parse_file(file_path)
    embeddings = get_embeddings(file_path, "mistral", para)
    
    prompt = input('What do you want to know ?')
    prompt_embedding = ollama.embeddings(model="mistral", prompt=prompt)['embedding']
    similar = find_similar(prompt_embedding, embeddings)[:5]
    
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(para[item[1]] for item in similar),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])

if __name__ == "__main__":
    main()
