
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
