IndustryResearchAgent
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
