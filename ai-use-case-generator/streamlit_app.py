# streamlit_app.py
import streamlit as st
import asyncio
import logging
from typing import Dict, TypedDict, Annotated
from datetime import datetime
from langgraph.graph import StateGraph, END
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
        workflow = StateGraph(
            state_schema=WorkflowState,
        )

        async def research(state: WorkflowState) -> Dict:
            analysis = await self.research_agent.research_company(state["company"])
            return {"analysis": analysis}

        async def generate_use_cases(state: WorkflowState) -> Dict:
            use_cases = await self.use_case_gen.generate_use_cases(state["analysis"])
            return {"use_cases": use_cases}

        async def collect_resources(state: WorkflowState) -> Dict:
            resources = await self.resource_collector.collect_resources(state["use_cases"])
            return {"resources": resources}

        async def generate_report(state: WorkflowState) -> Dict:
            report = self.report_gen.generate_report(
                state["company"],
                state["analysis"], 
                state["use_cases"],
                state["resources"]
            )
            return {"report": report, "__end__": True}

        # Add nodes
        workflow.add_node("research", research)
        workflow.add_node("generate_use_cases", generate_use_cases)
        workflow.add_node("collect_resources", collect_resources) 
        workflow.add_node("generate_report", generate_report)

        # Add edges
        workflow.add_edge("research", "generate_use_cases")
        workflow.add_edge("generate_use_cases", "collect_resources")
        workflow.add_edge("collect_resources", "generate_report")
        
        # Set entry/exit
        workflow.set_entry_point("research")
        workflow.set_finish_point("generate_report")

        return workflow

    async def run_workflow(self, company: str) -> str:
        try:
            initial_state = {
                "company": company,
                "timestamp": datetime.now(),
                "analysis": {},
                "use_cases": [],
                "resources": {},
                "report": ""
            }
            
            result = await self.graph.ainvoke(initial_state)
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