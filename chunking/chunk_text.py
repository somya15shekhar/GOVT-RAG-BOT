import nltk
nltk.download('punkt',quiet = True)

from nltk.tokenize import sent_tokenize
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks with specified size and overlap.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping characters between chunks.
        
    Returns:
        list: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip()) # save current chunk  
            # Start a new chunk with overlap
            current_chunk = current_chunk[-overlap:] + ' ' + sentence #from prev chunk add a few chars as overlap to next sentence
            #can do Minor change in above line to avoid breaking words in overlap
            current_length = len(current_chunk)
        else:
            current_chunk += ' ' + sentence if current_chunk else sentence
            current_length += sentence_length   
            
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks   
    