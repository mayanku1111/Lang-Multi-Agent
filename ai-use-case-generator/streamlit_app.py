import streamlit as st
import asyncio
import logging
from typing import Dict
from datetime import datetime
from langchain.graphs import StateGraph
from config import CONFIG

from agents.research_agent import EnhancedResearchAgent
from agents.use_case_generator import EnhancedUseCaseGenerator
from agents.resource_collector import EnhancedResourceCollector
from markdown_generator import MarkdownGenerator

class WorkflowManager:
    def __init__(self, config: Dict):
        self.research_agent = EnhancedResearchAgent(config)
        self.use_case_gen = EnhancedUseCaseGenerator(config)
        self.resource_collector = EnhancedResourceCollector(config)
        self.report_gen = MarkdownGenerator()
        self.graph = self._create_workflow_graph()

    def _create_workflow_graph(self) -> StateGraph:

        graph = StateGraph()
        
        graph.add_node("research", self.research_agent.research_company)
        graph.add_node("generate_use_cases", self.use_case_gen.generate_use_cases)
        graph.add_node("collect_resources", self.resource_collector.collect_resources)
        graph.add_node("generate_report", self.report_gen.generate_report)
        
        graph.add_edge("research", "generate_use_cases")
        graph.add_edge("generate_use_cases", "collect_resources")
        graph.add_edge("collect_resources", "generate_report")
        
        return graph

    async def run_workflow(self, company: str) -> str:
        try:
            result = await self.graph.arun({
                "company": company,
                "timestamp": datetime.now()
            })
            
            return result["report"]
            
        except Exception as e:
            logging.error(f"Workflow error: {str(e)}")
            raise

def main():
    st.title("AI Use Case Generator")
    
    workflow_manager = WorkflowManager(CONFIG)
    
    company = st.text_input("Enter Company or Industry Name:")
    
    if st.button("Generate Analysis"):
        with st.spinner("Analyzing..."):
            try:
                report = asyncio.run(workflow_manager.run_workflow(company))
                
                st.markdown(report)
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"{company}_analysis.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()