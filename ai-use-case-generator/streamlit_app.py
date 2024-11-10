import streamlit as st
from agents.research_agent import ResearchAgent
from agents.use_case_generator import UseCaseGenerator
from agents.resource_collector import ResourceCollector
from markdown_generator import MarkdownGenerator

def main():
    st.title("AI Use Case Generator")
    
    # Input section
    company = st.text_input("Enter Company or Industry Name:")
    
    if st.button("Generate Analysis"):
        with st.spinner("Researching company and industry..."):
            # Initialize agents
            research_agent = ResearchAgent()
            use_case_gen = UseCaseGenerator()
            resource_collector = ResourceCollector()
            
            # Execute workflow
            try:
                # Company research
                analysis = research_agent.search_and_analyze(company)
                st.success("Company analysis completed!")
                
                # Generate use cases
                use_cases = use_case_gen.generate_use_cases(analysis)
                st.success("Use cases generated!")
                
                # Collect resources
                resources = resource_collector.collect_resources(use_cases)
                st.success("Resources collected!")
                
                # Generate markdown report
                report = MarkdownGenerator.generate_report(
                    company=company,
                    analysis=analysis,
                    use_cases=use_cases,
                    resources=resources
                )
                
                # Display results
                st.markdown(report)
                
                # Download button for markdown report
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"{company}_analysis.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()