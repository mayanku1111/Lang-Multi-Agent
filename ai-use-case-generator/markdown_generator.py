# markdown_generator.py

from typing import Dict, List, Union
from agents import CompanyAnalysis

class MarkdownGenerator:
    def generate_report(self, company: str, analysis: Union[Dict, CompanyAnalysis], use_cases: List[Dict], resources: Dict) -> str:
        # Remove @staticmethod to access instance methods
        analysis_dict = analysis if isinstance(analysis, dict) else analysis.to_dict()
        
        markdown = f"""
# Market Research & AI Use Case Analysis: {company}
## Company Analysis
Industry: {analysis_dict.get('industry', '')}
Business Model: {analysis_dict.get('business_model', '')}
Key Products: {', '.join(analysis_dict.get('key_products', []))}
Market Position: {analysis_dict.get('market_position', '')}
Competitors: {', '.join(analysis_dict.get('competitors', []))}
AI Readiness Score: {analysis_dict.get('ai_readiness', 0.0)}

## Industry Trends & Market Position
{analysis_dict.get('trends', '')}

## AI/ML Use Cases
"""
        for use_case in use_cases:
            markdown += self._format_use_case(use_case, resources)
        return markdown

    def _format_use_case(self, use_case: Dict, resources: Dict) -> str:
        markdown = f"""
### {use_case['title']}
- **Description**: {use_case['description']}
- **Business Impact**: {use_case['impact']}
- **Complexity**: {use_case['complexity']}
- **Timeline**: {use_case['timeline']}
#### Resources & Datasets
"""
        use_case_resources = resources.get(use_case['title'], {})
        return markdown + self._format_resources(use_case_resources)

    def _format_resources(self, resources: Dict) -> str:
        markdown = ""
        if resources.get('datasets'):
            markdown += "\n##### Datasets\n"
            for dataset in resources['datasets']:
                markdown += f"- [{dataset['title']}]({dataset['url']}) - {dataset['description']}\n"
        
        if resources.get('documentation'):
            markdown += "\n##### Documentation & References\n"
            for doc in resources['documentation']:
                markdown += f"- [{doc['title']}]({doc['url']}) - {doc['description']}\n"

        if resources.get('github'):
            markdown += "\n##### GitHub Repositories\n"
            for repo in resources['github']:
                markdown += f"- [{repo['title']}]({repo['url']}) - {repo['description']}\n"
    
        return markdown        


    def save_resources(self, company: str, resources: Dict, output_path: str = "resources.md"):
        with open(output_path, "w") as f:
            f.write(f"# AI Resources for {company}\n\n")
            for use_case_title, use_case_resources in resources.items():
                f.write(f"## {use_case_title}\n")
                f.write(self._format_resources(use_case_resources))
                f.write("\n---\n\n")