"""
BenchmarkProcessor Class
Handles benchmark processing and metrics computation - implementation from 2_Model_Benchmark.py
"""

import evaluate
import pandas as pd
import re
import string
from typing import List, Dict
import json


class BenchmarkProcessor:
    """
    Benchmark processing and metrics computation.
    Based on 2_Model_Benchmark.py functions.
    """
    
    def __init__(self):
        """Initialize BenchmarkProcessor with SQuAD metric"""
        self.squad_metric = evaluate.load("squad")
    
    def process_benchmark_file(self, file, file_extension: str) -> pd.DataFrame:
        """
        Process uploaded benchmark file (CSV or JSON).
        Implementation from 2_Model_Benchmark.py file upload handling.
        """
        if file_extension == 'csv':
            df = pd.read_csv(file)
            
            if 'id' not in df.columns:
                df['id'] = [str(i) for i in range(len(df))]
            
            df['answer'] = df['answer'].apply(
                lambda x: x.split('|||') if isinstance(x, str) and '|||' in x 
                else ([x] if isinstance(x, str) else x)
            )
            
        elif file_extension == 'json':
            data = json.load(file)
            
            if 'data' in data:
                rows = []
                for article in data['data']:
                    for para in article['paragraphs']:
                        context = para['context']
                        for qa in para['qas']:
                            all_answers = [ans['text'] for ans in qa['answers']]
                            answer_starts = [ans['answer_start'] for ans in qa['answers']]
                            rows.append({
                                'id': qa.get('id', str(len(rows))),
                                'context': context,
                                'question': qa['question'],
                                'answer': all_answers,
                                'answer_start': answer_starts
                            })
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(data)
                if 'id' not in df.columns:
                    df['id'] = [str(i) for i in range(len(df))]
                if 'answer' in df.columns:
                    df['answer'] = df['answer'].apply(
                        lambda x: x if isinstance(x, list) else [x]
                    )
        
        return df
    
    def compute_metrics_squad(self, predictions: List[Dict], 
                             references: List[Dict]) -> Dict[str, float]:
        """
        Compute F1 and EM using library evaluate (same as training).
        Implementation from 2_Model_Benchmark.py compute_metrics_squad function.
        """
        results = self.squad_metric.compute(
            predictions=predictions,
            references=references
        )
        
        return {
            'f1': results['f1'],
            'exact_match': results['exact_match']
        }
    
    def normalize_answer(self, s: str) -> str:
        """
        Lower text and remove punctuation, articles and extra whitespace.
        Implementation from 2_Model_Benchmark.py normalize_answer function.
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def download_results(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to CSV for download"""
        return df.to_csv(index=False)
