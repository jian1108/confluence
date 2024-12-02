from typing import List, Dict, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.model = model
        openai.api_key = api_key
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def enhance_answer(self, question: str, original_answer: str, 
                      context: Optional[str] = None) -> Dict:
        """Enhance the answer using LLM"""
        try:
            # Prepare the prompt
            prompt = self._create_enhancement_prompt(question, original_answer, context)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """
                        You are a helpful assistant that improves answers to questions.
                        Make the answers more clear, concise, and well-structured.
                        Keep the technical accuracy but make it more understandable.
                        Add relevant examples when appropriate.
                        Format the response in markdown when useful.
                    """},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            enhanced_answer = response.choices[0].message.content
            
            return {
                'enhanced_answer': enhanced_answer,
                'original_answer': original_answer,
                'enhancement_type': 'full',
                'model_used': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error enhancing answer: {str(e)}")
            return {
                'enhanced_answer': original_answer,
                'original_answer': original_answer,
                'enhancement_type': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def generate_summary(self, qa_pairs: List[Dict]) -> str:
        """Generate a summary of multiple Q&A pairs"""
        try:
            qa_text = "\n\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in qa_pairs[:5]  # Limit to prevent token overflow
            ])
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the main points from these Q&A pairs."},
                    {"role": "user", "content": qa_text}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary generation failed"

    def _create_enhancement_prompt(self, question: str, answer: str, 
                                 context: Optional[str] = None) -> str:
        """Create prompt for answer enhancement"""
        prompt_parts = [
            f"Question: {question}\n",
            f"Original Answer: {answer}\n",
            "\nPlease enhance this answer by:"
            "1. Making it more clear and concise"
            "2. Adding structure if needed"
            "3. Including relevant examples"
            "4. Using markdown formatting when helpful"
        ]
        
        if context:
            prompt_parts.insert(2, f"Context: {context}\n")
            
        return "\n".join(prompt_parts)

    def validate_answer(self, question: str, answer: str) -> Dict:
        """Validate answer quality and accuracy"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """
                        Evaluate the answer quality and accuracy.
                        Check for:
                        - Completeness
                        - Accuracy
                        - Clarity
                        - Relevance to question
                    """},
                    {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return {
                'validation_result': response.choices[0].message.content,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            return {
                'validation_result': 'Validation failed',
                'error': str(e)
            } 