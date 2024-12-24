#Interface Implementation
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
