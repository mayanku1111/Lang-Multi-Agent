# resource_collector.py

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from langchain_community.tools import TavilySearchResults
from . import UseCase, Resource, ResourceType

class EnhancedResourceCollector:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        try:
            self.search_tool = TavilySearchResults(api_key=config.get("tavily_api_key"))
            self.github_token = config.get("github_api")
            self.kaggle_username = config.get("kaggle_username")
            self.kaggle_key = config.get("kaggle_api")
            self.session = None
            self.max_retries = 3
            self.retry_delay = 1
        except Exception as e:
            self.logger.error(f"Failed to initialize ResourceCollector: {str(e)}")
            raise

    async def _init_session(self):
        """Initialize aiohttp session with retry logic."""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
        except Exception as e:
            self.logger.error(f"Failed to initialize session: {str(e)}")
            raise

    async def _cleanup_session(self):
        """Safely cleanup aiohttp session."""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.error(f"Error closing session: {str(e)}")
            finally:
                self.session = None

    async def collect_resources(self, use_cases: List[UseCase]) -> List[Resource]:
        """Collect resources for use cases with improved error handling."""
        if not use_cases:
            self.logger.warning("No use cases provided")
            return []

        all_resources = []
        
        try:
            await self._init_session()
            
            for use_case in use_cases:
                try:
                    # Gather resources concurrently for each use case
                    resource_tasks = [
                        self._search_datasets(use_case),
                        self._search_documentation(use_case),
                        self._search_github(use_case)
                    ]
                    
                    results = await asyncio.gather(*resource_tasks, return_exceptions=True)
                    
                    # Process results and handle exceptions
                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Error gathering resources: {str(result)}")
                            continue
                        if isinstance(result, list):
                            all_resources.extend(result)
                            
                except Exception as e:
                    self.logger.error(f"Error processing use case {use_case.title}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Critical error in collect_resources: {str(e)}")
            raise
        finally:
            await self._cleanup_session()
            
        return all_resources

    async def _search_with_retry(self, query: str) -> List[Dict]:
        """Search with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await self.search_tool.ainvoke({"query": query})
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Search failed after {self.max_retries} attempts: {str(e)}")
                    return []
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return []

    async def _validate_resource(self, resource: Resource) -> bool:
        """Validate resource URL with retry logic."""
        for attempt in range(self.max_retries):
            try:
                if not self.session or self.session.closed:
                    await self._init_session()
                    
                async with self.session.head(
                    resource.url, 
                    allow_redirects=True, 
                    timeout=10
                ) as response:
                    return response.status == 200
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.debug(f"Resource validation failed for {resource.url}: {str(e)}")
                    return False
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return False

    async def _search_datasets(self, use_case: UseCase) -> List[Resource]:
        """Search for datasets with improved error handling."""
        try:
            query = f"{use_case.title} dataset data repository"
            results = await self._search_with_retry(query)
            
            validated_resources = []
            for result in results:
                try:
                    resource = Resource(
                        type=ResourceType.DATASET,
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("snippet", ""),
                        relevance_score=result.get("relevance_score", 0.0),
                        use_case_id=use_case.title
                    )
                    if await self._validate_resource(resource):
                        validated_resources.append(resource)
                except Exception as e:
                    self.logger.error(f"Error processing dataset result: {str(e)}")
                    continue
                    
            return validated_resources
        except Exception as e:
            self.logger.error(f"Error in _search_datasets: {str(e)}")
            return []

    async def _search_documentation(self, use_case: UseCase) -> List[Resource]:
        """Search for documentation with improved error handling."""
        try:
            query = f"{use_case.title} tutorial documentation guide"
            results = await self._search_with_retry(query)
            
            validated_resources = []
            for result in results:
                try:
                    resource = Resource(
                        type=ResourceType.DOCUMENTATION,
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("snippet", ""),
                        relevance_score=result.get("relevance_score", 0.0),
                        use_case_id=use_case.title
                    )
                    if await self._validate_resource(resource):
                        validated_resources.append(resource)
                except Exception as e:
                    self.logger.error(f"Error processing documentation result: {str(e)}")
                    continue
                    
            return validated_resources
        except Exception as e:
            self.logger.error(f"Error in _search_documentation: {str(e)}")
            return []

    async def _search_github(self, use_case: UseCase) -> List[Resource]:
        """Search GitHub repositories with improved error handling."""
        if not self.github_token:
            return []
            
        try:
            query = f"{use_case.title} {use_case.description}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            if not self.session or self.session.closed:
                await self._init_session()
                
            async with self.session.get(
                f"https://api.github.com/search/repositories?q={query}",
                headers=headers,
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    resources = []
                    for repo in data.get("items", [])[:5]:
                        try:
                            resource = Resource(
                                type=ResourceType.GITHUB_REPO,
                                title=repo["name"],
                                url=repo["html_url"],
                                description=repo.get("description", ""),
                                relevance_score=0.8,  # Default score for GitHub repos
                                use_case_id=use_case.title
                            )
                            resources.append(resource)
                        except Exception as e:
                            self.logger.error(f"Error processing GitHub repo: {str(e)}")
                            continue
                    return resources
                else:
                    self.logger.error(f"GitHub API returned status: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"GitHub search error: {str(e)}")
            return []