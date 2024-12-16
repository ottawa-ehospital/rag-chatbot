from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
import fitz
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = 'OPEN_API_KEY'  # Replace with your actual API key

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

chat = ChatOpenAI(model='gpt-3.5-turbo')
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Process PDF and store in Pinecone
    text = process_pdf(filepath)
    store_in_pinecone(text, filename)
    
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    augmented_query = augment_prompt(query)
    response = generate_response(augmented_query)
    
    return jsonify({
        'response': response,
        'augmentedQuery': augmented_query
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    print("Test endpoint called")   
    return jsonify({'message': 'Hello from the RAG Chatbot API!'})

def process_pdf(filepath):
    text = ""
    doc = fitz.open(filepath)
    for page in doc:
        text += page.get_text()
    return text

def store_in_pinecone(text, filename):
    index_name = 'rag-chatbot-index'
    if index_name not in pc.list_indexes():
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='dotproduct',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as e:
            print(f"Error creating index: {e}")
            pass
    
    index = pc.Index(index_name)
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings = embed_model.embed_documents([text])
    
    metadata = {'text': text, 'source': 'PDF Document', 'title': filename}
    doc_id = f"pdf-{filename.split('.')[0]}"
    index.upsert(vectors=[(doc_id, embeddings[0], metadata)])

def augment_prompt(query):
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    index = pc.Index('rag-chatbot-index')
    vectorstore = LangchainPinecone(index, embed_model, "text")
    
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    
    return f"""Using the contexts below, answer the query.

Contexts:
{source_knowledge}

Query: {query}"""

def generate_response(augmented_query):
    prompt = HumanMessage(content=augmented_query)
    res = chat.invoke([SystemMessage(content="You are a helpful assistant.")] + [prompt])
    return res.content

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Use Heroku's dynamic port or fallback to 5001
    app.run(debug=True, host='0.0.0.0', port=port)

