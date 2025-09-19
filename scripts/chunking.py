def chunk_text(text, chunk_size=200, chunk_overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks

if __name__=="__main__":
    import os
    data_dir = "../data/"
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                text = f.read().replace("\n", " ")
            chunks = chunk_text(text)
            print(f"{filename}: {len(chunks)} chunks")
