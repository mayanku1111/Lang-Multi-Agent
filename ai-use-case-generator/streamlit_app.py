import streamlit as st
import asyncio
import logging
from typing import Dict, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph
from config import CONFIG

from agents.research_agent import EnhancedResearchAgent
from agents.use_case_generator import EnhancedUseCaseGenerator 
from agents.resource_collector import EnhancedResourceCollector
from markdown_generator import MarkdownGenerator

class WorkflowState(TypedDict):
    company: str
    timestamp: datetime
    analysis: Dict
    use_cases: list
    resources: Dict
    report: str

class WorkflowManager:
    def __init__(self, config: Dict):
        self.research_agent = EnhancedResearchAgent(config)
        self.use_case_gen = EnhancedUseCaseGenerator(config)
        self.resource_collector = EnhancedResourceCollector(config)
        self.report_gen = MarkdownGenerator()
        self.graph = self._create_workflow_graph()

    def _create_workflow_graph(self) -> StateGraph:
        # Define the workflow graph with state schema
        graph = StateGraph(state_schema=WorkflowState)

        # Add nodes with state transformations
        async def research(state):
            analysis = await self.research_agent.research_company(state["company"])
            return {"analysis": analysis}

        async def generate_use_cases(state):
            use_cases = await self.use_case_gen.generate_use_cases(state["analysis"])
            return {"use_cases": use_cases}

        async def collect_resources(state):
            resources = await self.resource_collector.collect_resources(state["use_cases"])
            return {"resources": resources}

        async def generate_report(state):
            report = self.report_gen.generate_report(
                state["company"],
                state["analysis"],
                state["use_cases"],
                state["resources"]
            )
            return {"report": report}

        # Add nodes
        graph.add_node("research", research)
        graph.add_node("generate_use_cases", generate_use_cases)
        graph.add_node("collect_resources", collect_resources)
        graph.add_node("generate_report", generate_report)

        # Add edges
        graph.add_edge("research", "generate_use_cases")
        graph.add_edge("generate_use_cases", "collect_resources")
        graph.add_edge("collect_resources", "generate_report")
        graph.set_entry_point("research")
        graph.set_finish_point("generate_report")

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