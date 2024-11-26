import joblib
import time
from django.shortcuts import render
from sentence_transformers import SentenceTransformer, util

data = joblib.load('embeddings/product_data_embeddings.joblib')
titles = data['titles']
prices = data['prices']
ratings = data['ratings']
colors = data['colors']
links = data['links']
embeddings = data['embeddings']
model = SentenceTransformer('LaBSE')

def search_product(query, threshold=0.3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings)[0]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    top_hits = [hit for hit in hits if hit['score'] >= threshold]
    return top_hits

def search_product(query, threshold=0.3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings)[0]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    top_hits = [hit for hit in hits if hit['score'] >= threshold]
    return top_hits

def index(request):
    results = []
    query = None
    
    if request.method == 'POST':
        query = request.POST.get('query', '')
        if query:
            start_time = time.time()
            hits = search_product(query)
            end_time = time.time()
            
            results = [
                {
                    'title': titles[hit['corpus_id']],
                    'price': prices[hit['corpus_id']],
                    'rating': ratings[hit['corpus_id']],
                    'color': colors[hit['corpus_id']],
                    'link': links[hit['corpus_id']],
                    'score': hit['score']
                }
                for hit in hits[:4]  
            ]
            elapsed_time = end_time - start_time
            print(f"Search completed in {elapsed_time:.3f} seconds!")
            
    return render(request, 'index/home.html', {'results': results, 'query': query})