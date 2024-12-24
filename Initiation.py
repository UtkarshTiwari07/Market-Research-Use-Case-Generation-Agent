import warnings
warnings.filterwarnings('ignore')

import os
import json
from typing import List, Dict
from datetime import datetime
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from tenacity import retry, wait_exponential, stop_after_attempt
from crewai import Agent, Task
from crewai_tools import SerperDevTool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from IPython.display import JSON, display, HTML, clear_output
import ipywidgets as widgets
from pydantic import BaseModel, Field
import google.generativeai as genai

# Set API keys
os.environ["GEMINI_API_KEY"] = " "
os.environ["SERPER_API_KEY"] = " "

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define Gemini LLM wrapper
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

class GeminiLLM:
    def __init__(self):
        self.model_name = "gemini-2.0-flash-exp"
        self.generation_config = generation_config

    def predict(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Error interacting with Gemini: {e}")
            raise

# Initialize Gemini LLM
llm = GeminiLLM()
# Cell 1: Improved search implementation
from pydantic import BaseModel, Field

class EnhancedSerperTool(SerperDevTool):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields

    def search(self, query: str) -> str:
        """Enhanced search with retries"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                results = self.run(query=query)
                
                if not results:
                    raise Exception("No results returned")
                
                # Format the results
                formatted_results = []
                if isinstance(results, list):
                    for idx, result in enumerate(results[:5]):
                        if isinstance(result, dict):
                            formatted_result = f"\nResult {idx + 1}:\n"
                            formatted_result += f"Title: {result.get('title', 'N/A')}\n"
                            formatted_result += f"Snippet: {result.get('snippet', 'N/A')}\n"
                            formatted_result += f"Link: {result.get('link', 'N/A')}\n"
                            formatted_results.append(formatted_result)
                
                if formatted_results:
                    return "\n".join(formatted_results)
                else:
                    return "No relevant results found"
                
            except Exception as e:
                print(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"Search failed after {max_retries} attempts"

# Initialize tools
search_tool = EnhancedSerperTool()

# Create backup search function
def backup_search(self, query: str) -> str:
    try:
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': 5
        }
        
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            results = response.json()
            formatted_results = []
            
            if 'organic' in results:
                for idx, result in enumerate(results['organic'][:5]):
                    formatted_result = f"\nResult {idx + 1}:\n"
                    formatted_result += f"Title: {result.get('title', 'N/A')}\n"
                    formatted_result += f"Snippet: {result.get('snippet', 'N/A')}\n"
                    formatted_result += f"Link: {result.get('link', 'N/A')}\n"
                    formatted_results.append(formatted_result)
            
            return "\n".join(formatted_results) if formatted_results else "No relevant results found"
        else:
            return f"Search failed with status code: {response.status_code}"
            
    except Exception as e:
        print(f"Backup search error: {str(e)}")
        return "Backup search failed"
# Define search schema
class SearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query to execute",
        examples=["company overview business profile"]
    )

# Create Tool instance with schema
search_tool_with_schema = Tool(
    name="Internet Search",
    func=search_tool.search,  # Use the search method
    description="Search the internet for information",
    args_schema=SearchSchema
)

# Define the LLM response function with retries
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_llm_response(prompt: str) -> str:
    try:
        # Call the Gemini LLM
        response = llm.predict(prompt)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        raise

print("Initialization complete: Search tool and Gemini LLM ready.")
