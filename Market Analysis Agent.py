#Market Analysis Agent
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
