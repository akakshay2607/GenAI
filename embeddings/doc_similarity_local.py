from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #384 dimensional vector


documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'who is bumrah?'
embeded_query = embeddings.embed_query(query)

docs = embeddings.embed_documents(documents)

scores = cosine_similarity([embeded_query],docs)[0]

index,score = sorted(list(enumerate(list(scores))),key=lambda x:x[1])[-1]
print(documents[index],score)