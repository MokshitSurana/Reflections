from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from pydantic import BaseModel
import pandas as pd
import gzip
import torch
import string
from gensim.summarization import summarize
from transformers import ElectraTokenizer, ElectraForMaskedLM
from fastpunct import FastPunct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

fastpunct = FastPunct()
top_k = 10
app = FastAPI()

origins = ["http://localhost",
"http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

clf = pickle.load(open("sentiment.pkl",'rb'))
vectorizer = pickle.load(open("transform.pkl", "rb"))
df = pickle.load(open("medium_articles_cleaned.pkl", "rb"))
with gzip.open('sim.pklz', 'rb') as ifp:
    sim = pickle.load(ifp)

electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

@app.get("/api/home")
async def Home():
    return {"message": "Hello World"}

    
class SentimentModel(BaseModel):
    comments: List[str]

class RecommendModel(BaseModel):
    index: int

class NextWordModel(BaseModel):
    text: str

class SummerizerModel(BaseModel):
    text: str
    
class CorrectorModel(BaseModel):
    text: str
    
class NewDataModel(BaseModel):
    data: List[str]

class NewDataModel(BaseModel):
    author: str
    tags: List[str]
    content: str
    title: str
    timestamp: str
    url: str
    index: int


@app.post("/api/sentiment")
async def Sentiment(body: SentimentModel):
    result = []
    for c in body.comments:
        p = clf.predict(vectorizer.transform([c]))[0]
        if(p == 1):
            result.append(1)
        else:
            result.append(0)
    return {"result": result}

@app.post("/api/recommend")
async def Recommend(body: RecommendModel):
    result = recommend(body.index)
    res = {'result': []}
    for i in result:
        res["result"].append(i.item())
    return res


def recommend(index):
    df = pickle.load(open("medium_articles_cleaned.pkl", "rb"))
    sim = pickle.load(gzip.open('sim.pklz', 'rb'))
    ind = df[df['index'] == index].index[0]
    distances = sorted(
        list(enumerate(sim[ind])), reverse=True, key=lambda x: x[1])
    rec = [df.iloc[i[0]]['index'] for i in distances[1:11]]
    return rec

# ------ Next word suggestion ------

@app.post("/api/nextword")
async def NextWord(body: NextWordModel):
    text = body.text + ' <mask>'
    result = get_next_predictions(text)
    print(result)
    return result

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = f'{string.punctuation}[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_next_predictions(text_sentence, top_clean=5):
    # ========================= ELECTRA =================================
    input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = electra_model(input_ids)[0]
    electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    words = electra.split('\n')
    return { 'result': words}


# ------ Summarizer ------
@app.post("/api/summarizer")
async def Summarizer(body: SummerizerModel):
    result = summarize(body.text, 0.14)
    result = result.replace('\n', ' ')
    return {"result": result}

# ------ Corrector ------
@app.post("/api/corrector")
async def Corrector(body: CorrectorModel):
    result = fastpunct.punct(body.text, correct=True)
    return {"result": result}

def vectorize(text, maxx_features):
    vectorizer = TfidfVectorizer(max_features=maxx_features, stop_words='english')
    X = vectorizer.fit_transform(text)
    return X


@app.post("/api/new_data_recommender")
def new_data_recommender(body: NewDataModel):
    global df
    comb = body.author + ' ' + " ".join(body.tags) + ' ' + body.content + ' ' + body.title
    new_data = [body.index, body.title, body.content, body.url, body.author, body.timestamp, body.tags, comb]
    df = df.append(pd.Series(new_data, index=df.columns), ignore_index=True)
    pickle.dump(df, open('medium_articles_cleaned.pkl', 'wb'))
    text = df['comb'].values
    X = vectorize(text, 2 ** 12)
    pca = PCA(n_components=0.95, random_state=42)
    X_reduced= pca.fit_transform(X.toarray())
    sim = cosine_similarity(X_reduced)
    with gzip.open('sim.pklz', 'wb') as ofp:
        pickle.dump(sim, ofp)
    df = pd.read_pickle('medium_articles_cleaned.pkl')
    sim = pickle.load(gzip.open('sim.pklz', 'rb'))
    ind = df[df['index'] == body.index].index[0]
    distances = sorted(
        list(enumerate(sim[ind])), reverse=True, key=lambda x: x[1])
    rec = [df.iloc[i[0]]['index'] for i in distances[1:11]]
    res = {'result': []}
    for i in rec:
        res["result"].append(i.item())
    return res
