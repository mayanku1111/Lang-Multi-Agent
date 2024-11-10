# resource_collector.py

from typing import Dict, List, Optional
from langchain_community.tools import TavilySearchResults
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

@dataclass
class UseCase:
    title: str
    description: str
    impact: str
    complexity: str
    timeline: str

class ResourceType(Enum):
    DATASET = "dataset"
    DOCUMENTATION = "documentation"
    GITHUB_REPO = "github_repo"
    RESEARCH_PAPER = "research_paper"

@dataclass
class Resource:
    type: ResourceType
    title: str
    url: str
    description: str
    relevance_score: float
    use_case_id: str

class EnhancedResourceCollector:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.search_tool = TavilySearchResults(api_key=config["tavily_api_key"])
        self.github_token = config.get("github_api")
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = 3
        self.retry_delay = 1

    async def collect_resources(self, use_cases: List[Dict]) -> Dict[str, Dict]:
        """Collect resources for each use case."""
        resources = {}
        try:
            for use_case in use_cases:
                use_case_obj = UseCase(
                    title=use_case["title"],
                    description=use_case["description"],
                    impact=use_case["impact"],
                    complexity=use_case["complexity"],
                    timeline=use_case["timeline"]
                )
                resources[use_case_obj.title] = {
                    "datasets": await self._search_datasets(use_case_obj),
                    "documentation": await self._search_documentation(use_case_obj),
                    "github": await self._search_github(use_case_obj)
                }
        except Exception as e:
            self.logger.error(f"Error collecting resources: {str(e)}")
        finally:
            if self.session and not self.session.closed:
                await self.session.close()
        return resources