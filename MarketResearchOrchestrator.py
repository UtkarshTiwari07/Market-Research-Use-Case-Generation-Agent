 
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
