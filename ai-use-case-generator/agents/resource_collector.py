from langchain_openai import ChatOpenAI
from typing import Dict, List
import asyncio
import aiohttp
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from . import UseCase, Resource  

class EnhancedResourceCollector:
    def __init__(self, config: Dict):
        self.search_tool = TavilySearchResults()
        self.github_api = config["github_api"]
        self.kaggle_api = config["kaggle_api"]

    async def collect_resources(self, use_cases: List[UseCase]) -> Dict[str, List[Resource]]:
        resources = {}
        
        for use_case in use_cases:
            resources[use_case.title] = await asyncio.gather(
                self._search_datasets(use_case),
                self._search_github(use_case),
                self._search_documentation(use_case)
            )
            
        return resources

    async def _validate_resource(self, resource: Resource) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(resource.url) as response:
                    return response.status == 200
        except Exception:
            return False