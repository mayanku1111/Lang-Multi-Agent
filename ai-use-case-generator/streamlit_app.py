import streamlit as st
import asyncio
import logging
from typing import Dict, TypedDict, Literal
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
    analysis: Dict
    use_cases: list
    resources: Dict
    report: str

class WorkflowManager:
    def __init__(self, config: Dict):
        """Initialize workflow manager with configuration."""
        self.checkpointer = MemorySaver()
        self.research_agent = EnhancedResearchAgent(config)
        self.use_case_gen = EnhancedUseCaseGenerator(config)
        self.resource_collector = EnhancedResourceCollector(config)
        self.report_gen = MarkdownGenerator()
        self.graph = self._create_workflow_graph()

    def _create_workflow_graph(self) -> StateGraph:
        async def research(state: WorkflowState) -> Dict:
            analysis = await self.research_agent.research_company(state["company"])
            return {"analysis": analysis}

        async def generate_use_cases(state: WorkflowState) -> Dict:
            use_cases = await self.use_case_gen.generate_use_cases(state["analysis"])
            return {"use_cases": use_cases}

        async def collect_resources(state: WorkflowState) -> Dict:
            resources = await self.resource_collector.collect_resources(state["use_cases"])
            return {"resources": resources}

        def generate_report(state: WorkflowState) -> Dict:
            report = self.report_gen.generate_report(
                state["company"],
                state["analysis"],
                state["use_cases"],
                state["resources"]
            )
            return {"report": report}

        def should_continue(state: WorkflowState) -> Literal["generate_use_cases", "collect_resources", "generate_report", END]:
            if not state.get("analysis"):
                return "generate_use_cases"
            elif not state.get("use_cases"):
                return "collect_resources"
            elif not state.get("resources"):
                return "generate_report"
            return END

        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("research", research)
        workflow.add_node("generate_use_cases", generate_use_cases)
        workflow.add_node("collect_resources", collect_resources)
        workflow.add_node("generate_report", generate_report)

        # Set entry point and edges
        workflow.set_entry_point("research")
        
        workflow.add_conditional_edges(
            "research",
            should_continue,
            {
                "generate_use_cases": "generate_use_cases",
                "collect_resources": "collect_resources",
                "generate_report": "generate_report",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "generate_use_cases",
            should_continue,
            {
                "collect_resources": "collect_resources",
                "generate_report": "generate_report",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "collect_resources",
            should_continue,
            {
                "generate_report": "generate_report",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "generate_report",
            should_continue,
            {END: END}
        )

        return workflow.compile(checkpointer=self.checkpointer)

    async def run_workflow(self, company: str) -> str:
        try:
            initial_state: WorkflowState = {
                "messages": [],
                "company": company,
                "timestamp": datetime.now(),
                "analysis": {},
                "use_cases": [],
                "resources": {},
                "report": ""
            }
            
            config = {
                "configurable": {
                    "thread_id": company,
                },
                "recursion_limit": 50
            }
            
            result = await self.graph.ainvoke(initial_state, config=config)
            
            if isinstance(result, dict) and "report" in result:
                return result["report"]
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())