from typing import Dict, List, Optional
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from textblob import TextBlob

class QACategorizer:
    def __init__(self):
        # Predefined categories and their keywords
        self.category_patterns = {
            'Technical': [
                r'how to|error|bug|issue|failed|doesn\'t work|setup|install|configure|deploy',
                r'api|endpoint|service|database|server|code|programming|development',
                r'integration|authentication|permission|access|token|credential'
            ],
            'Process': [
                r'process|workflow|procedure|steps|guide|tutorial|documentation',
                r'policy|standard|requirement|compliance|regulation|rule',
                r'approve|review|submit|request|create|update|modify'
            ],
            'Administrative': [
                r'account|user|group|team|role|permission|access',
                r'license|subscription|payment|billing|cost|price',
                r'admin|administrator|manage|configuration|setting'
            ],
            'Product': [
                r'feature|functionality|capability|product|service',
                r'version|release|update|upgrade|migration|compatibility',
                r'roadmap|planning|timeline|schedule|milestone'
            ],
            'Support': [
                r'help|support|assistance|contact|reach out',
                r'ticket|incident|request|issue|problem|resolution',
                r'escalate|priority|urgent|emergency|critical'
            ]
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def categorize_rule_based(self, question: str) -> List[str]:
        """Categorize question using rule-based approach"""
        question = question.lower()
        categories = []
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    categories.append(category)
                    break
        
        return list(set(categories)) or ['Uncategorized']

    def extract_topics(self, questions: List[str], num_topics: int = 5) -> List[str]:
        """Extract topics using ML clustering"""
        if len(questions) < num_topics:
            return ['General'] * len(questions)
            
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(questions)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_topics, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms for each cluster
        cluster_centers = kmeans.cluster_centers_
        feature_names = self.vectorizer.get_feature_names_out()
        
        topics = []
        for cluster_idx in clusters:
            center = cluster_centers[cluster_idx]
            top_terms_idx = center.argsort()[-3:][::-1]  # Get top 3 terms
            topic = " / ".join([feature_names[i] for i in top_terms_idx])
            topics.append(topic)
            
        return topics

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def categorize_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Categorize QA pairs using multiple approaches"""
        # Extract questions for ML-based categorization
        questions = [qa['question'] for qa in qa_pairs]
        ml_topics = self.extract_topics(questions)
        
        categorized_pairs = []
        for idx, qa in enumerate(qa_pairs):
            # Rule-based categorization
            rule_based_categories = self.categorize_rule_based(qa['question'])
            
            # Sentiment analysis
            question_sentiment = self.analyze_sentiment(qa['question'])
            answer_sentiment = self.analyze_sentiment(qa['answer'])
            
            # Enhance QA pair with categorization
            enhanced_qa = {
                **qa,
                'categories': rule_based_categories,
                'ml_topic': ml_topics[idx],
                'metadata': {
                    'question_sentiment': question_sentiment,
                    'answer_sentiment': answer_sentiment,
                    'complexity': self._estimate_complexity(qa['question'], qa['answer'])
                }
            }
            categorized_pairs.append(enhanced_qa)
        
        return categorized_pairs

    def _estimate_complexity(self, question: str, answer: str) -> str:
        """Estimate complexity of QA pair"""
        # Simple heuristics for complexity estimation
        answer_length = len(answer.split())
        technical_terms = len(re.findall(r'api|code|technical|configure|implementation|architecture', 
                                       question.lower() + answer.lower()))
        
        if answer_length > 200 or technical_terms > 3:
            return 'Advanced'
        elif answer_length > 100 or technical_terms > 1:
            return 'Intermediate'
        else:
            return 'Basic'

    def generate_category_summary(self, qa_pairs: List[Dict]) -> Dict:
        """Generate summary statistics for categorized QA pairs"""
        summary = {
            'category_distribution': defaultdict(int),
            'topic_distribution': defaultdict(int),
            'complexity_distribution': defaultdict(int),
            'sentiment_analysis': {
                'positive_questions': 0,
                'negative_questions': 0,
                'neutral_questions': 0
            }
        }
        
        for qa in qa_pairs:
            # Count categories
            for category in qa['categories']:
                summary['category_distribution'][category] += 1
            
            # Count topics
            summary['topic_distribution'][qa['ml_topic']] += 1
            
            # Count complexity
            summary['complexity_distribution'][qa['metadata']['complexity']] += 1
            
            # Count sentiment
            polarity = qa['metadata']['question_sentiment']['polarity']
            if polarity > 0.1:
                summary['sentiment_analysis']['positive_questions'] += 1
            elif polarity < -0.1:
                summary['sentiment_analysis']['negative_questions'] += 1
            else:
                summary['sentiment_analysis']['neutral_questions'] += 1
        
        return dict(summary) 