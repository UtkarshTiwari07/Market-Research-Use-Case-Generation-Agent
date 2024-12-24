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
os.environ["GEMINI_API_KEY"] = "  "
os.environ["SERPER_API_KEY"] = "  "

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

# Cell 2: Updated IndustryResearchAgent
class IndustryResearchAgent(Agent):
    def __init__(self):
         
        super().__init__(
            role="Industry Research Expert",
            goal="Research and analyze industry/company information comprehensively",
            backstory="Expert in industry analysis and market research with extensive experience in corporate intelligence and market analysis. Skilled at gathering and synthesizing information from multiple sources.",
            tools=[search_tool_with_schema],
            llm=llm,
            verbose=True
        )

    def execute_task(self, task) -> Dict:
        try:
            company_name = self._extract_company_name(task.description)
            print(f"Researching company: {company_name}")

            search_results = {}
            search_queries = {
                "company_overview": f"{company_name} company profile business overview recent news",
                "industry_sector": f"{company_name} industry sector market position competitors",
                "technology": f"{company_name} technology infrastructure digital transformation"
            }

            for key, query in search_queries.items():
                try:
                    print(f"Searching for {key}...")
                    result = self.tools[0].run(query=query)
                    if result and "Search failed" not in result:
                        search_results[key] = result
                    else:
                        print(f"No valid results for {key}")
                        search_results[key] = "No results found"
                    time.sleep(2)
                except Exception as e:
                    print(f"Search failed for {key}: {str(e)}")
                    search_results[key] = f"Search failed: {str(e)}"

            valid_results = [v for v in search_results.values() if "Search failed" not in v and "No results found" not in v]
            if valid_results:
                analysis_result = self._generate_analysis(company_name, search_results)
            else:
                analysis_result = "Insufficient data to generate analysis"

            return {
                "company_name": company_name,
                "industry_analysis": analysis_result,
                "raw_research": search_results
            }

        except Exception as e:
            print(f"Error in IndustryResearchAgent: {str(e)}")
            return {
                "error": str(e),
                "company_name": company_name if 'company_name' in locals() else "Unknown"
            }

    def _extract_company_name(self, description: str) -> str:
        try:
            cleaned_desc = description.lower()
            for prefix in ["research", "company", "about", "and provide"]:
                cleaned_desc = cleaned_desc.replace(prefix, "")
            company_name = cleaned_desc.strip().split()[0]
            return company_name
        except Exception:
            return "Unknown Company"

    def _generate_analysis(self, company_name: str, search_results: Dict) -> str:
        analysis_prompt = f"""
        Based on the following information about {company_name}, provide a comprehensive analysis:
        
        Company Overview: {search_results.get('company_overview', 'No data available')}
        Industry Sector: {search_results.get('industry_sector', 'No data available')}
        Technology: {search_results.get('technology', 'No data available')}
        
        Please provide a structured analysis covering:
        1. Company Overview and Background
        2. Industry Position and Market Share
        3. Technology Stack and Digital Initiatives
        4. Key Strengths and Competitive Advantages
        5. Challenges and Areas for Improvement
        6. Future Outlook and Opportunities
        
        Format the analysis in a clear, professional manner with section headers.
        """
        
        try:
            return get_llm_response(analysis_prompt)
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
# Market Analysis Agent
class MarketAnalysisAgent(Agent):
    def __init__(self):
         

        super().__init__(
            role="AI/ML Strategy Expert",  # Changed from name to role
            goal="Generate comprehensive AI/ML use cases based on industry analysis",
            backstory="Expert in AI/ML applications and industry trends with deep experience in digital transformation and technology implementation strategies.",
            tools=[search_tool_with_schema],
            llm=llm,
            verbose=True
        )

    def execute_task(self, task) -> Dict:
        try:
            industry_analysis = str(task.description) if task.description else ""
            print("Analyzing industry data for AI/ML opportunities...")

            try:
                # Generate initial use cases
                initial_use_cases = self._generate_initial_use_cases(industry_analysis)
                
                # Generate technical analysis
                technical_analysis = self._generate_technical_analysis(initial_use_cases)
                
                # Generate implementation roadmap
                implementation_roadmap = self._generate_implementation_roadmap(
                    initial_use_cases, technical_analysis
                )
                
                # Generate ROI analysis
                roi_analysis = self._generate_roi_analysis(initial_use_cases)

                return {
                    "use_cases": initial_use_cases,
                    "technical_analysis": technical_analysis,
                    "implementation_roadmap": implementation_roadmap,
                    "roi_analysis": roi_analysis,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            except RetryError as e:
                print(f"Rate limit error: {str(e)}")
                return {
                    "error": "Rate limit exceeded",
                    "partial_analysis": industry_analysis[:200] + "..."
                }

        except Exception as e:
            print(f"Error in MarketAnalysisAgent: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def _generate_initial_use_cases(self, industry_analysis: str) -> str:
        prompt = f"""
        Based on this industry analysis, identify specific AI/ML use cases:
        {industry_analysis}
        
        For each use case, provide:
        1. Title
        2. Description
        3. Implementation Requirements
        4. Expected Benefits
        5. Priority Level (High/Medium/Low)
        6. Estimated Timeline
        
        Format each use case clearly with headers and bullet points.
        """
        return get_llm_response(prompt)

    def _generate_technical_analysis(self, use_cases: str) -> str:
        prompt = f"""
        Analyze technical requirements for these use cases:
        {use_cases}
        
        For each use case, provide detailed specifications for:
        1. Required AI/ML Models and Algorithms
        2. Data Requirements and Sources
        3. Infrastructure and Computing Needs
        4. Integration Requirements
        5. Technical Skills Required
        6. Potential Technical Challenges
        
        Format the analysis with clear sections for each use case.
        """
        return get_llm_response(prompt)

    def _generate_implementation_roadmap(self, use_cases: str, technical_analysis: str) -> str:
        prompt = f"""
        Create a detailed implementation roadmap based on:
        Use Cases: {use_cases}
        Technical Analysis: {technical_analysis}
        
        Include for each phase:
        1. Timeline and Major Milestones
        2. Required Resources and Team Composition
        3. Dependencies and Prerequisites
        4. Risk Mitigation Strategies
        5. Success Metrics and KPIs
        
        Organize the roadmap in clear phases with specific timelines.
        """
        return get_llm_response(prompt)

    def _generate_roi_analysis(self, use_cases: str) -> str:
        prompt = f"""
        Provide a comprehensive ROI analysis for these use cases:
        {use_cases}
        
        For each use case, analyze:
        1. Implementation Costs (Development, Infrastructure, Training)
        2. Expected Benefits (Quantitative and Qualitative)
        3. Timeline to Value
        4. Risk Factors
        5. Break-even Analysis
        6. Long-term Value Proposition
        
        Present the analysis in a structured format with clear metrics.
        """
        return get_llm_response(prompt)
 

# Resource type patterns
RESOURCE_TYPE_PATTERNS = {
    'GitHub Repository': ['github.com', 'gitlab.com', 'bitbucket.org'],
    'Dataset': ['kaggle.com', 'huggingface.co/datasets', '.csv', '.json', '.xml', '.parquet'],
    'Tutorial': ['tutorial', 'guide', 'how-to', 'learn'],
    'Documentation': ['.pdf', '.md', '.txt', 'docs.', 'documentation'],
    'API': ['api.', '/api/', 'endpoint'],
    'Research Paper': ['arxiv.org', 'paper', 'research', '.pdf'],
    'Tool/Library': ['tool', 'library', 'framework', 'sdk']
}

class ResourceCollectionAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Resource Research Expert",  # Specify the role
            goal="Collect relevant datasets and resources for AI/ML implementations",  # Specify the goal
            backstory=(
                "Expert in finding and evaluating AI/ML resources, datasets, and implementation guides. "
                "Specialized in identifying high-quality technical resources and practical implementation materials."
            ),
            tools=[search_tool_with_schema],  # Specify the search tool
            llm=llm,  # Specify the LLM instance
            verbose=True  # Enable verbose logging
        )

    def execute_task(self, task) -> Dict:
        try:
            use_cases_text = task.description
            print("Starting resource collection...")

            # Step 1: Extract use cases
            use_case_titles = self._extract_use_cases(use_cases_text)
            if not use_case_titles:
                raise ValueError("No use cases extracted. Ensure the task description contains valid content.")

            # Step 2: Collect resources for each use case
            resources = self._collect_resources(use_case_titles)

            # Step 3: Generate implementation recommendations
            recommendations = self._generate_recommendations(resources) if resources else "No resources found to generate recommendations."

            return {
                "resource_collection": resources,
                "implementation_recommendations": recommendations,
                "summary": f"Collected resources for {len(resources)} use cases",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Error in ResourceCollectionAgent: {str(e)}")
            return {
                "error": str(e),
                "resource_collection": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def _extract_use_cases(self, text: str) -> List[str]:
        """Extract use case titles from analysis text."""
        extraction_prompt = f"""
        Extract the main AI/ML use cases from the following analysis. 
        Return only the titles/main topics of the use cases.

        Analysis:
        {text}
        """
        try:
            # Use the LLM to generate the response
            use_case_titles = get_llm_response(extraction_prompt).split('\n')
            return [title.strip() for title in use_case_titles if title.strip()]
        except Exception as e:
            print(f"Error extracting use cases: {str(e)}")
            return []

    def _collect_resources(self, use_case_titles: List[str]) -> List[Dict]:
        """Collect resources for each use case."""
        resources = []
        for use_case in use_case_titles:
            print(f"Processing use case: {use_case}")
            resources.append(self._collect_use_case_resources(use_case))
        return resources

    def _collect_use_case_resources(self, use_case: str) -> Dict:
        """Collect resources for a specific use case."""
        use_case_resources = {
            "use_case": use_case,
            "implementation_resources": {
                "datasets": [],
                "github_repos": [],
                "tutorials": [],
                "additional_resources": [],
            },
        }

        search_categories = {
            "github_repos": f"github repository {use_case} implementation AI ML",
            "datasets": f"dataset {use_case} machine learning AI",
            "tutorials": f"tutorial implementation {use_case} AI ML",
        }

        for category, query in search_categories.items():
            try:
                print(f"Searching {category} for {use_case}...")
                search_result = self.tools[0].run(query=query)
                if search_result:
                    analyzed_results = self._analyze_search_results(use_case, search_result)
                    use_case_resources["implementation_resources"][category] = analyzed_results
                else:
                    print(f"No valid results found for {category}")
            except Exception as e:
                print(f"Search failed for {category}: {str(e)}")

        return use_case_resources

    def _analyze_search_results(self, use_case: str, search_result: str) -> List[Dict]:
        """Analyze search results using LLM."""
        analysis_prompt = f"""
        Analyze these search results for {use_case}:
        {search_result}

        Extract and list relevant:
        1. URLs
        2. Resource names
        3. Brief descriptions

        Format each item as follows:
        - URL: [URL]
        - Name: [Resource Name]
        - Description: [Brief Description]
        """
        try:
            analyzed_results = get_llm_response(analysis_prompt)
            parsed_results = []
            for line in analyzed_results.split("\n"):
                if line.startswith("- URL:"):
                    url = line.split("[URL]")[1].strip(" []")
                    parsed_results.append({
                        "url": url,
                        "name": self._extract_resource_name(url),
                        "description": "Generated description for the resource." if "Description:" not in line else line.split("Description:")[1].strip(),
                        "type": self._determine_resource_type(url),
                        "added_date": datetime.now().strftime("%Y-%m-%d")
                    })
            return parsed_results
        except Exception as e:
            print(f"Error analyzing search results: {str(e)}")
            return []

    def _format_resource(self, resource: str) -> Dict:
        """Format a resource entry with metadata."""
        try:
            url = None
            for part in resource.split():
                if part.startswith("http"):
                    url = part
                    break

            return {
                "name": self._extract_resource_name(resource) if url else resource,
                "url": url,
                "type": self._determine_resource_type(resource),
                "description": resource.replace(url, "").strip() if url else resource,
                "added_date": datetime.now().strftime("%Y-%m-%d"),
            }
        except Exception as e:
            print(f"Error formatting resource: {str(e)}")
            return {
                "name": "Unknown Resource",
                "url": None,
                "type": "Unknown",
                "description": "Error processing resource",
                "added_date": datetime.now().strftime("%Y-%m-%d"),
            }

    def _extract_resource_name(self, resource: str) -> str:
        """Extract a readable name from the resource."""
        try:
            if "http" in resource.lower():
                parts = resource.split("/")
                name = parts[-1] if parts[-1] else parts[-2]
                name = name.split("?")[0].replace("-", " ").replace("_", " ").strip()
                return name.title()
            return resource[:100] if len(resource) > 100 else resource
        except Exception:
            return "Unnamed Resource"

    def _determine_resource_type(self, resource: str) -> str:
        """Determine the type of resource."""
        resource_lower = resource.lower()
        for resource_type, patterns in RESOURCE_TYPE_PATTERNS.items():
            if any(pattern in resource_lower for pattern in patterns):
                return resource_type
        return "Other"

    def _generate_recommendations(self, resources: List[Dict]) -> str:
        """Generate implementation recommendations based on collected resources."""
        recommendations_prompt = f"""
        Based on these collected resources:
        {json.dumps(resources, indent=2)}

        Provide implementation recommendations covering:
        1. Technical requirements and prerequisites
        2. Implementation steps and best practices
        3. Potential challenges and solutions
        4. Integration considerations
        5. Resource utilization guidelines

        Format recommendations by use case.
        """
        try:
            return get_llm_response(recommendations_prompt)
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return "Unable to generate recommendations."

 
class MarketResearchOrchestrator:
    def __init__(self):
        self.industry_agent = IndustryResearchAgent()
        self.market_agent = MarketAnalysisAgent()
        self.resource_agent = ResourceCollectionAgent()

    def _format_results_for_display(self, results: Dict) -> Dict:
        """
        Format the final results for display or further processing.
        """
        try:
            # Format results with clean structure
            formatted_results = {
                "Company Name": results["metadata"]["company_name"],
                "Timestamp": results["metadata"]["timestamp"],
                "Status": results["metadata"]["status"],
                "Industry Analysis": results["industry_analysis"],
                "Use Cases": results["use_cases"],
                "Technical Details": results["technical_details"],
                "Implementation Plan": results["implementation_plan"],
                "Resources": results["resources"]
            }
            return formatted_results
        except KeyError as e:
            print(f"Error formatting results: Missing key {e}")
            return results  # Return the original results in case of failure

    def run_analysis(self, company_name: str) -> Dict:
        try:
            print(f"Starting analysis for {company_name}...")

            # Step 1: Industry Research
            industry_task = Task(
                description=f"Research company {company_name} and provide industry analysis",
                expected_output="Detailed industry analysis",
                agent=self.industry_agent
            )

            print("Conducting industry research...")
            industry_results = self.industry_agent.execute_task(industry_task)

            if "error" in industry_results:
                raise Exception(f"Industry research failed: {industry_results['error']}")

            industry_analysis_text = str(industry_results.get('industry_analysis', ''))

            # Step 2: Market Analysis
            market_task = Task(
                description=industry_analysis_text,
                expected_output="AI/ML use cases",
                agent=self.market_agent
            )

            print("Generating AI/ML use cases...")
            use_case_results = self.market_agent.execute_task(market_task)
            use_cases_text = str(use_case_results.get('use_cases', ''))

            # Step 3: Resource Collection
        #if use_cases_text and "error" not in use_case_results:
            resource_task = Task(
                    description=use_cases_text,
                    expected_output="Implementation resources",
                    agent=self.resource_agent
                )

            print("Collecting implementation resources...")
            resource_results = self.resource_agent.execute_task(resource_task)
            #else:
                #resource_results = {"resources": []}

            # Final results
            final_results = {
                "metadata": {
                    "company_name": company_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Success"
                },
                "industry_analysis": industry_analysis_text,
                "use_cases": use_cases_text,
                "technical_details": use_case_results.get('technical_details', ''),
                "implementation_plan": use_case_results.get('implementation_plan', ''),
                "resources": resource_results.get('resources', [])
            }

            # Save the results to a JSON and CSV file
            self._save_results_to_file(company_name, final_results)

            return self._format_results_for_display(final_results)

        except Exception as e:
            error_message = f"Error in orchestrator: {str(e)}"
            print(error_message)
            return {
                "error": error_message,
                "status": "Failed",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def _save_results_to_file(self, company_name: str, results: Dict):
        """
        Save the results to local JSON and CSV files.
        """
        json_filename = f"{company_name.replace(' ', '_')}_analysis.json"
        csv_filename = f"{company_name.replace(' ', '_')}_analysis.csv"
        try:
            # Save as JSON
            with open(json_filename, 'w') as json_file:
                json.dump(results, json_file, indent=2)
            print(f"Results saved to {json_filename}")

            # Save as CSV
            formatted_csv_data = self._prepare_csv_data(results)
            with open(csv_filename, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Field", "Value"])
                for key, value in formatted_csv_data.items():
                    writer.writerow([key, value])
            print(f"Results saved to {csv_filename}")

        except Exception as e:
            print(f"Error saving results to file: {e}")

    def _prepare_csv_data(self, results: Dict) -> Dict:
        """
        Flatten results to prepare for CSV storage.
        """
        csv_data = {
            "Company Name": results["metadata"]["company_name"],
            "Timestamp": results["metadata"]["timestamp"],
            "Status": results["metadata"]["status"],
            "Industry Analysis": results["industry_analysis"],
            "Use Cases": results["use_cases"],
            "Technical Details": results["technical_details"],
            "Implementation Plan": results["implementation_plan"]
        }
        # Add resources as a single field in CSV
        resources = results.get("resources", [])
        if resources:
            csv_data["Resources"] = "; ".join([str(resource) for resource in resources])
        else:
            csv_data["Resources"] = "None"
        return csv_data

# Interface Implementation
def create_analysis_interface():
    # Create layout
    layout = widgets.Layout(width='500px', margin='10px')
    
    # Create and display widgets
    title = widgets.HTML(value="<h2>AI Use Case Generator</h2>")
    display(title)
    
    description = widgets.HTML(
        value="<p>Enter a company name to generate AI/ML use cases and analysis.</p>"
    )
    display(description)
    
    input_box = widgets.Text(
        description='Company:',
        placeholder='Enter company name here',
        layout=layout
    )
    display(input_box)
    
    button = widgets.Button(
        description='Analyze',
        button_style='primary',
        layout=layout
    )
    display(button)
    
    status = widgets.HTML(value="")
    display(status)
    
    output = widgets.Output(layout=widgets.Layout(
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0'
    ))
    display(output)
    
    def on_button_click(b):
        with output:
            clear_output()
            if not input_box.value:
                status.value = '<p style="color: red;">Please enter a company name</p>'
                return
            
            status.value = '<p style="color: blue;">Analysis in progress... This may take a few minutes.</p>'
            try:
                # Initialize orchestrator
                orchestrator = MarketResearchOrchestrator()
                
                # Run analysis with error handling
                try:
                    results = orchestrator.run_analysis(input_box.value)
                    
                    # Save results
                    filename = f"{input_box.value.replace(' ', '_')}_analysis.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Display results sections
                    display(HTML("<h3>Analysis Results:</h3>"))
                    display(JSON(results))
                    
                    status.value = f'<p style="color: green;">Analysis completed! Results saved to {filename}</p>'
                    
                except Exception as e:
                    status.value = f'<p style="color: red;">Analysis Error: {str(e)}</p>'
                    print(f"Analysis error details: {str(e)}")
                
            except Exception as e:
                status.value = f'<p style="color: red;">System Error: {str(e)}</p>'
                print(f"System error details: {str(e)}")
    
    button.on_click(on_button_click)

# Run the interface
create_analysis_interface()
