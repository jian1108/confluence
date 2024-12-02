from flask import Flask, render_template, request, jsonify
from crawler.confluence_crawler import ConfluenceCrawler
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

crawler = ConfluenceCrawler(
    url=os.getenv('CONFLUENCE_URL'),
    username=os.getenv('CONFLUENCE_USERNAME'),
    api_token=os.getenv('CONFLUENCE_API_TOKEN')
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    enhance = request.json.get('enhance', True)
    results = crawler.search(query, enhance_results=enhance)
    return jsonify(results)

@app.route('/crawl', methods=['POST'])
def crawl():
    space_key = request.json.get('space_key')
    force_refresh = request.json.get('force_refresh', False)
    
    try:
        qa_pairs = crawler.crawl_space(space_key, force_refresh)
        return jsonify({
            'status': 'success',
            'message': f'Crawled {len(qa_pairs)} QA pairs'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/enhance', methods=['POST'])
def enhance():
    """Enhance a specific answer"""
    data = request.json
    enhancement = crawler.llm_processor.enhance_answer(
        data['question'],
        data['answer'],
        data.get('context')
    )
    return jsonify(enhancement)

if __name__ == '__main__':
    app.run(debug=True) 