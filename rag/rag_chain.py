from together import Together

class RAGChain:
    def __init__(self, retriever, api_key, model = "meta-llama/Llama-3.3-70B-Instruct-Turbo", max_tokens=300):
        self.client = Together(api_key=api_key)
        self.retriever = retriever
        self.model = model
        self.max_tokens = max_tokens

    def generate_prompt(self, question, contexts):
        context_text = "\n\n---\n\n".join(contexts)
        prompt = f"""You are an assistant for Indian government schemes. Use only the following context to answer the question. If the answer is not in the context, reply "I don't know."

Context:
{context_text}

Question:
{question}

Answer:"""
        return prompt

    def answer_question(self, question, top_k):
        query_embedding = self.retriever.model.encode(question)
        contexts = self.retriever.search(query_embedding, top_k)
        prompt = self.generate_prompt(question, contexts)

          # Add this logging:
        print("\n🔍 Retrieved Chunks:")
        for i, chunk in enumerate(contexts, 1):
            print(f"--- Chunk {i} ---")
            print(chunk[:500])  # print first 500 chars to avoid flooding
            print()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
