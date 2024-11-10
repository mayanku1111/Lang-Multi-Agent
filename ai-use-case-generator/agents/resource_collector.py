from langchain_openai import ChatOpenAI


class ResourceCollector:
    def __init__(self):
        self.search_tool = TavilySearchResults()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        
    def collect_resources(self, use_cases: List[Dict]) -> Dict:
        resources = {}
        
        for use_case in use_cases:
            # Search for relevant datasets and resources
            search_query = f"dataset github kaggle {use_case['title']} {use_case['description']}"
            results = self.search_tool.invoke(search_query)
            
            resources[use_case['title']] = {
                'datasets': [r for r in results if 'kaggle.com' in r['url'] or 'github.com' in r['url']],
                'documentation': [r for r in results if '.org' in r['url'] or '.io' in r['url']]
            }
            
        return resources