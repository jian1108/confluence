from typing import Dict, List
from collections import defaultdict
import logging
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

class ExtractionStats:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize statistics containers
        self.reset_stats()
        
        # Set up statistics logger
        self.stats_logger = logging.getLogger('extraction_stats')
        self.setup_logger()

    def setup_logger(self):
        """Set up a dedicated logger for extraction statistics"""
        stats_handler = logging.FileHandler(
            self.log_dir / f'extraction_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        stats_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.stats_logger.addHandler(stats_handler)
        self.stats_logger.setLevel(logging.INFO)

    def reset_stats(self):
        """Reset all statistics counters"""
        self.stats = {
            'total_pages_processed': 0,
            'total_qa_pairs': 0,
            'qa_pairs_per_page': defaultdict(int),
            'extraction_patterns': defaultdict(int),
            'failed_pages': [],
            'empty_pages': [],
            'processing_times': [],
            'qa_length_stats': {
                'question_lengths': [],
                'answer_lengths': []
            },
            'category_stats': defaultdict(int),
            'complexity_stats': defaultdict(int),
            'section_stats': defaultdict(int)
        }

    def log_page_processing(self, page_info: Dict, qa_pairs: List[Dict], 
                          processing_time: float, pattern: str):
        """Log statistics for a processed page"""
        self.stats['total_pages_processed'] += 1
        pairs_count = len(qa_pairs)
        self.stats['total_qa_pairs'] += pairs_count
        self.stats['qa_pairs_per_page'][page_info['id']] = pairs_count
        self.stats['extraction_patterns'][pattern] += pairs_count
        self.stats['processing_times'].append(processing_time)

        if pairs_count == 0:
            self.stats['empty_pages'].append(page_info['id'])

        # Log QA pair lengths
        for qa in qa_pairs:
            self.stats['qa_length_stats']['question_lengths'].append(
                len(qa['question'].split())
            )
            self.stats['qa_length_stats']['answer_lengths'].append(
                len(qa['answer'].split())
            )
            
            # Log category and complexity stats
            for category in qa.get('categories', []):
                self.stats['category_stats'][category] += 1
            self.stats['complexity_stats'][qa.get('metadata', {}).get('complexity', 'Unknown')] += 1
            
            # Log section statistics
            section = qa.get('reference', {}).get('section')
            if section:
                self.stats['section_stats'][section] += 1

    def log_extraction_failure(self, page_info: Dict, error: str):
        """Log failed page extraction"""
        self.stats['failed_pages'].append({
            'page_id': page_info['id'],
            'page_title': page_info['title'],
            'error': str(error)
        })

    def generate_summary(self) -> Dict:
        """Generate a comprehensive statistics summary"""
        summary = {
            'general_stats': {
                'total_pages_processed': self.stats['total_pages_processed'],
                'total_qa_pairs': self.stats['total_qa_pairs'],
                'average_qa_per_page': self.stats['total_qa_pairs'] / max(1, self.stats['total_pages_processed']),
                'failed_pages_count': len(self.stats['failed_pages']),
                'empty_pages_count': len(self.stats['empty_pages']),
                'average_processing_time': sum(self.stats['processing_times']) / max(1, len(self.stats['processing_times']))
            },
            'qa_stats': {
                'avg_question_length': sum(self.stats['qa_length_stats']['question_lengths']) / 
                                     max(1, len(self.stats['qa_length_stats']['question_lengths'])),
                'avg_answer_length': sum(self.stats['qa_length_stats']['answer_lengths']) / 
                                   max(1, len(self.stats['qa_length_stats']['answer_lengths'])),
                'extraction_patterns': dict(self.stats['extraction_patterns']),
                'category_distribution': dict(self.stats['category_stats']),
                'complexity_distribution': dict(self.stats['complexity_stats']),
                'section_distribution': dict(self.stats['section_stats'])
            },
            'issues': {
                'failed_pages': self.stats['failed_pages'],
                'empty_pages': self.stats['empty_pages']
            }
        }
        return summary

    def save_stats_report(self):
        """Save detailed statistics report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON summary
        summary = self.generate_summary()
        with open(self.log_dir / f'extraction_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV reports
        self._save_csv_reports(timestamp)
        
        # Log summary
        self.stats_logger.info(f"Statistics Summary:\n{json.dumps(summary['general_stats'], indent=2)}")

    def _save_csv_reports(self, timestamp: str):
        """Save detailed CSV reports"""
        # QA pairs per page
        pd.DataFrame(list(self.stats['qa_pairs_per_page'].items()), 
                    columns=['page_id', 'qa_count']).to_csv(
            self.log_dir / f'qa_per_page_{timestamp}.csv', index=False
        )
        
        # Category distribution
        pd.DataFrame(list(self.stats['category_stats'].items()), 
                    columns=['category', 'count']).to_csv(
            self.log_dir / f'category_distribution_{timestamp}.csv', index=False
        ) 