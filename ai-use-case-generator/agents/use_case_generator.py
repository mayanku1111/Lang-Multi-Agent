from chromadb import PersistentClient
from langchain_openai import ChatOpenAI


class UseCaseGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.db = PersistentClient(path="./chroma_db")
        self.collection = self.db.get_or_create_collection("use_cases")
    
    def generate_use_cases(self, company_analysis: Dict) -> List[Dict]:
        prompt = f"""
        Based on the following company analysis, generate specific AI/ML use cases:
        {company_analysis['analysis']}
        
        For each use case provide:
        1. Title
        2. Description
        3. Business Impact
        4. Implementation Complexity (High/Medium/Low)
        5. Required Resources
        6. Estimated Timeline
        """
        
        response = self.llm.invoke(prompt)
        
        # Store use cases in ChromaDB
        self.collection.add(
            documents=[str(response.content)],
            metadatas=[{"company": company_analysis['company'], "type": "use_cases"}],
            ids=[f"{company_analysis['company']}_use_cases"]
        )
        
        return response.content