class MarkdownGenerator:
    @staticmethod
    def generate_report(company: str, analysis: Dict, use_cases: List[Dict], resources: Dict) -> str:
        markdown = f"""
# Market Research & AI Use Case Analysis: {company}

## Company Analysis
{analysis['analysis']}

## Industry Trends & Market Position
{analysis.get('trends', '')}

## AI/ML Use Cases

"""
        
        for use_case in use_cases:
            markdown += f"""
### {use_case['title']}
- **Description**: {use_case['description']}
- **Business Impact**: {use_case['impact']}
- **Complexity**: {use_case['complexity']}
- **Timeline**: {use_case['timeline']}

#### Resources & Datasets
"""
            
            use_case_resources = resources.get(use_case['title'], {})
            if use_case_resources.get('datasets'):
                markdown += "\nDatasets:\n"
                for dataset in use_case_resources['datasets']:
                    markdown += f"- [{dataset['title']}]({dataset['url']})\n"
            
            if use_case_resources.get('documentation'):
                markdown += "\nDocumentation & Guides:\n"
                for doc in use_case_resources['documentation']:
                    markdown += f"- [{doc['title']}]({doc['url']})\n"
                    
        return markdown