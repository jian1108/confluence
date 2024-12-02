from atlassian import Confluence
from bs4 import BeautifulSoup, ParserError
import os
from typing import List, Dict, Optional
import logging
import html
from urllib.parse import unquote
import re
import json
import hashlib
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

# Add cache-related imports
import pickle
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HTMLParsingError(Exception):
    """Custom exception for HTML parsing errors"""
    pass

class CacheManager:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "confluence_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS page_cache (
                    page_id TEXT PRIMARY KEY,
                    space_key TEXT,
                    content BLOB,
                    qa_pairs BLOB,
                    last_modified TEXT,
                    cache_date TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS space_metadata (
                    space_key TEXT PRIMARY KEY,
                    last_crawled TEXT,
                    page_count INTEGER
                )
            """)

    def get_cached_page(self, page_id: str) -> Optional[Dict]:
        """Retrieve cached page content and QA pairs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content, qa_pairs, cache_date FROM page_cache WHERE page_id = ?",
                (page_id,)
            )
            result = cursor.fetchone()
            
            if result:
                content, qa_pairs, cache_date = result
                return {
                    'content': pickle.loads(content),
                    'qa_pairs': pickle.loads(qa_pairs),
                    'cache_date': cache_date
                }
        return None

    def cache_page(self, page_id: str, space_key: str, content: str, 
                  qa_pairs: List[Dict], last_modified: str):
        """Cache page content and extracted QA pairs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO page_cache 
                (page_id, space_key, content, qa_pairs, last_modified, cache_date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    page_id,
                    space_key,
                    pickle.dumps(content),
                    pickle.dumps(qa_pairs),
                    last_modified,
                    datetime.now().isoformat()
                )
            )

    def update_space_metadata(self, space_key: str, page_count: int):
        """Update space metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO space_metadata 
                (space_key, last_crawled, page_count)
                VALUES (?, ?, ?)
                """,
                (space_key, datetime.now().isoformat(), page_count)
            )

    def clear_cache(self, space_key: Optional[str] = None):
        """Clear cache for a specific space or all spaces"""
        with sqlite3.connect(self.db_path) as conn:
            if space_key:
                conn.execute("DELETE FROM page_cache WHERE space_key = ?", (space_key,))
                conn.execute("DELETE FROM space_metadata WHERE space_key = ?", (space_key,))
            else:
                conn.execute("DELETE FROM page_cache")
                conn.execute("DELETE FROM space_metadata")

# Add to imports
from .qa_categorizer import QACategorizer
from .stats_logger import ExtractionStats
import time
from .vector_store import VectorStore
from .llm_processor import LLMProcessor

class ConfluenceCrawler:
    def __init__(self, url: str, username: str, api_token: str, 
                 openai_api_key: str,
                 cache_dir: str = ".cache",
                 cache_ttl: int = 24):  # TTL in hours
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
        self.parsing_errors = []
        self.cache_manager = CacheManager(cache_dir)
        self.cache_ttl = timedelta(hours=cache_ttl)
        self.categorizer = QACategorizer()
        self.stats = ExtractionStats()
        self.vector_store = VectorStore(cache_dir=cache_dir)
        self.llm_processor = LLMProcessor(openai_api_key)
        
    @lru_cache(maxsize=100)
    def get_page_last_modified(self, page_id: str) -> str:
        """Get page last modified date with memory caching"""
        page_info = self.confluence.get_page_by_id(page_id, expand='version')
        return page_info['version']['when']

    def should_update_cache(self, page_id: str, cached_date: str) -> bool:
        """Check if cache should be updated based on TTL and page modifications"""
        cache_datetime = datetime.fromisoformat(cached_date)
        if datetime.now() - cache_datetime > self.cache_ttl:
            # Check if page was modified since last cache
            last_modified = self.get_page_last_modified(page_id)
            return datetime.fromisoformat(last_modified) > cache_datetime
        return False

    def sanitize_html(self, content: str) -> str:
        """Sanitize HTML content before parsing"""
        try:
            # Decode HTML entities
            content = html.unescape(content)
            
            # Fix common Confluence macro issues
            content = content.replace('<ac:structured-macro', '<div class="structured-macro"')
            content = content.replace('</ac:structured-macro>', '</div>')
            
            # Remove CDATA sections while preserving content
            content = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', content, flags=re.DOTALL)
            
            # Fix unclosed tags (basic fix)
            common_tags = ['p', 'div', 'span', 'li', 'ul', 'ol', 'table', 'tr', 'td']
            for tag in common_tags:
                open_tags = content.count(f'<{tag}')
                close_tags = content.count(f'</{tag}>')
                if open_tags > close_tags:
                    content += f'</{tag}>' * (open_tags - close_tags)
            
            return content
        except Exception as e:
            logger.warning(f"Error during HTML sanitization: {str(e)}")
            return content

    def safe_parse_html(self, content: str) -> Optional[BeautifulSoup]:
        """Safely parse HTML content with multiple fallback parsers"""
        parsers = ['lxml', 'html.parser', 'html5lib']
        
        for parser in parsers:
            try:
                return BeautifulSoup(content, parser)
            except Exception as e:
                logger.warning(f"Parser {parser} failed: {str(e)}")
                continue
        
        raise HTMLParsingError("All HTML parsers failed")

    def extract_qa_from_page(self, content: str, page_info: Dict) -> List[Dict]:
        """Extract QA pairs with statistics logging"""
        start_time = time.time()
        try:
            qa_pairs = []
            
            # Pattern 1: FAQ sections
            faq_pairs = self._extract_faq_sections(content, page_info)
            if faq_pairs:
                self.stats.log_page_processing(
                    page_info, faq_pairs, 
                    time.time() - start_time, 
                    'faq_section'
                )
                qa_pairs.extend(faq_pairs)
            
            # Add other extraction patterns...
            
            return qa_pairs
            
        except Exception as e:
            self.stats.log_extraction_failure(page_info, str(e))
            raise

    def _extract_faq_sections(self, content: str, page_info: Dict) -> List[Dict]:
        """Extract FAQ sections with headers and paragraphs"""
        soup = self.safe_parse_html(content)
        qa_pairs = []
        
        # Create page reference
        page_reference = {
            'page_id': page_info['id'],
            'page_title': page_info['title'],
            'page_url': f"{self.confluence.url}/pages/viewpage.action?pageId={page_info['id']}",
            'space_key': page_info.get('space_key', ''),
            'last_modified': self.get_page_last_modified(page_info['id']),
            'section': None  # Will be populated with section heading if available
        }
        
        # Pattern 1: FAQ sections with headers and paragraphs
        faq_sections = soup.find_all(['h1', 'h2', 'h3'], string=lambda x: 'FAQ' in str(x).upper())
        for section in faq_sections:
            current_section = section.text.strip()
            current = section.find_next_sibling()
            
            while current and not (current.name in ['h1', 'h2', 'h3'] and 
                                 current.get('level', 0) <= section.get('level', 0)):
                if current.name in ['h4', 'h5', 'h6']:
                    question = current.text.strip()
                    answer_parts = []
                    answer_location = None
                    next_elem = current.find_next_sibling()
                    
                    while next_elem and next_elem.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        if next_elem.name in ['p', 'ul', 'ol', 'div']:
                            answer_parts.append(next_elem.text.strip())
                            if not answer_location:
                                # Store the location of the first answer part
                                answer_location = self._get_element_location(next_elem)
                        next_elem = next_elem.find_next_sibling()
                    
                    if answer_parts:
                        section_reference = {**page_reference, 'section': current_section}
                        if answer_location:
                            section_reference['anchor'] = answer_location
                        
                        qa_pairs.append({
                            'question': question,
                            'answer': ' '.join(answer_parts),
                            'reference': section_reference,
                            'context': current_section
                        })
                
                current = current.find_next_sibling()
        
        return qa_pairs

    def _get_element_location(self, element) -> Optional[str]:
        """Get the location (id or closest header) of an element"""
        # Check for element ID
        element_id = element.get('id')
        if element_id:
            return element_id
        
        # Look for closest header with ID
        prev_header = element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if prev_header and prev_header.get('id'):
            return prev_header.get('id')
        
        # Generate a hash of the content if no ID is found
        content_hash = hashlib.md5(element.text.encode()).hexdigest()[:8]
        return f"section-{content_hash}"

    def generate_answer_link(self, reference: Dict) -> str:
        """Generate a direct link to the answer"""
        base_url = reference['page_url']
        if reference.get('anchor'):
            return f"{base_url}#{reference['anchor']}"
        return base_url

    def crawl_space(self, space_key: str, force_refresh: bool = False) -> List[Dict]:
        """Crawl space and build vector store"""
        qa_pairs = super().crawl_space(space_key, force_refresh)
        
        # Build vector store
        if force_refresh or not self.vector_store.load(space_key):
            logger.info("Building vector store...")
            self.vector_store.add_qa_pairs(qa_pairs)
            self.vector_store.save(space_key)
        
        return qa_pairs
        
    def search(self, query: str, k: int = 5, enhance_results: bool = True) -> List[Dict]:
        """Search with LLM enhancement"""
        # Get initial results
        results = self.vector_store.search(query, k)
        
        if enhance_results:
            # Enhance each result with LLM
            for result in results:
                enhancement = self.llm_processor.enhance_answer(
                    query,
                    result['answer'],
                    result.get('context')
                )
                result['enhanced_answer'] = enhancement['enhanced_answer']
                result['enhancement_metadata'] = {
                    'type': enhancement['enhancement_type'],
                    'model': enhancement.get('model_used'),
                    'timestamp': enhancement['timestamp']
                }
                
                # Validate answer
                validation = self.llm_processor.validate_answer(
                    query,
                    enhancement['enhanced_answer']
                )
                result['validation'] = validation
        
        return results

    def process_qa_batch(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Process a batch of QA pairs with LLM"""
        processed_pairs = []
        
        for qa in qa_pairs:
            try:
                # Enhance answer
                enhancement = self.llm_processor.enhance_answer(
                    qa['question'],
                    qa['answer'],
                    qa.get('context')
                )
                
                # Add enhanced content
                qa['enhanced_answer'] = enhancement['enhanced_answer']
                qa['enhancement_metadata'] = {
                    'type': enhancement['enhancement_type'],
                    'model': enhancement.get('model_used'),
                    'timestamp': enhancement['timestamp']
                }
                
                processed_pairs.append(qa)
                
            except Exception as e:
                logger.error(f"Error processing QA pair: {str(e)}")
                processed_pairs.append(qa)  # Keep original if enhancement fails
                
        return processed_pairs

    def clear_cache(self, space_key: Optional[str] = None):
        """Clear the cache for a specific space or all spaces"""
        self.cache_manager.clear_cache(space_key)
        # Also clear the LRU cache for last_modified dates
        self.get_page_last_modified.cache_clear()

    def get_error_report(self) -> Dict:
        """Generate error report for the crawling process"""
        return {
            'parsing_errors': self.parsing_errors,
            'error_count': len(self.parsing_errors),
            'error_types': {}
        }