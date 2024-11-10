# streamlit_app.py
import streamlit as st
import asyncio
import logging
from typing import Dict, TypedDict, Literal, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from config import CONFIG

from agents.research_agent import EnhancedResearchAgent
from agents.use_case_generator import EnhancedUseCaseGenerator 
from agents.resource_collector import EnhancedResourceCollector
from markdown_generator import MarkdownGenerator

class WorkflowState(MessagesState, TypedDict):
    company: str
    timestamp: datetime
    analysis: Optional[Dict]
    use_cases: Optional[list]
    resources: Optional[Dict]
    report: Optional[str]

class WorkflowManager:
    def __init__(self, config: Dict):
        self.config = config
        self.research_agent = EnhancedResearchAgent(config)
        self.use_case_gen = EnhancedUseCaseGenerator(config)
        self.resource_collector = EnhancedResourceCollector(config)
        self.report_gen = MarkdownGenerator()
        self.graph = self._create_workflow_graph()
    
    def _create_workflow_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        
        async def research(state: WorkflowState) -> WorkflowState:
            state["analysis"] = await self.research_agent.research_company(state["company"])
            return state
            
        async def generate_use_cases(state: WorkflowState) -> WorkflowState:
            state["use_cases"] = await self.use_case_gen.generate_use_cases(state["analysis"])
            return state
            
        async def collect_resources(state: WorkflowState) -> WorkflowState:
            state["resources"] = await self.resource_collector.collect_resources(state["use_cases"])
            return state
            
        def generate_report(state: WorkflowState) -> WorkflowState:
            state["report"] = self.report_gen.generate_report(
                state["company"],
                state["analysis"],
                state["use_cases"],
                state["resources"]
            )
            return state

        workflow.add_node("research", research)
        workflow.add_node("generate_use_cases", generate_use_cases)
        workflow.add_node("collect_resources", collect_resources)
        workflow.add_node("generate_report", generate_report)

        workflow.set_entry_point("research")
        workflow.add_edge("research", "generate_use_cases")
        workflow.add_edge("generate_use_cases", "collect_resources")
        workflow.add_edge("collect_resources", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def run_workflow(self, company: str) -> str:
        try:
            initial_state = WorkflowState(
                messages=[],
                company=company,
                timestamp=datetime.now(),
                analysis=None,
                use_cases=None,
                resources=None,
                report=None
            )
            
            final_state = await self.graph.ainvoke(initial_state)
            
            if isinstance(final_state, dict) and final_state.get("report"):
                return final_state["report"]
            raise ValueError("Workflow failed to generate report")
            
        except Exception as e:
            logging.error(f"Workflow error: {str(e)}")
            raise

async def main():
    st.title("AI Use Case Generator")
    workflow_manager = WorkflowManager(CONFIG)
    company = st.text_input("Enter Company or Industry Name:")
    
    if st.button("Generate Analysis"):
        with st.spinner("Analyzing..."):
            try:
                report = await workflow_manager.run_workflow(company)
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
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())