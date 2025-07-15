import openai

class RAGChain:
    def __init__(self, retriever, openai_api_key, model="gpt-3.5-turbo", max_tokens=512):
        openai.api_key = openai_api_key  # Set the OpenAI API key
        self.retriever = retriever       # Retriever to fetch relevant documents
        self.model = model               # GPT model to use
        self.max_tokens = max_tokens     # Max tokens for the response

    def generate_prompt(self, question, contexts):
        # Join context chunks with separators
        context_text = "\n\n---\n\n".join(contexts)

        # Format the prompt with instructions, context, and the user's question
        prompt = f"""You are an expert assistant on Indian government schemes. Use the following context to answer the question. If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{question}

Answer:"""
        return prompt

    def answer_question(self, question, top_k=3):
        # Get vector embedding of the question
        query_embedding = self.retriever.model.encode(question)

        # Retrieve top_k relevant context chunks
        contexts = self.retriever.search(query_embedding, top_k)

        # Create the prompt for the language model
        prompt = self.generate_prompt(question, contexts)

        # Send prompt to OpenAI and get a response
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,  # Lower temperature = more factual
        )

        # Get the answer from the response
        # Note: OpenAI's response structure may vary, so I ensure to adapt this part as needed
        # - Take the first choice from the response
        # - Access the 'content' field (the actual answer)
        answer = response['choices'][0]['message']['content'].strip()
        return answer
