from textwrap import dedent
from typing import Any, Dict, List
from pydantic import create_model, Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class DataQualityAnalystModel(BaseModel):
    data_description: str = Field(description="Provide a detailed description of the keyword.")
    data_type: str = Field(description="Assign an appropriate Python data type to the keyword from the following options: int, float, str, or list.")

def create_data_quality_prompt(parser: JsonOutputParser) -> PromptTemplate:
    system_prompt = dedent("""
        You are a Data Quality Analyst with 20 years of professional experience.
        Your responsibilities include ensuring data quality, identifying the correct data types, and maintaining the Data Dictionary.

        Based on the provided keyword, answer the question. If the text does not contain a specific value, return null.
    """)

    return PromptTemplate(
        template=system_prompt + "\n{format_instructions}\nkeyword: {keyword}\n",
        input_variables=["keyword"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

def create_chain(keyword: str, prompt: PromptTemplate, parser: JsonOutputParser, llm: ChatOpenAI) -> Dict[str, Any]:
    chain = prompt | llm | parser
    return chain.invoke({"keyword": keyword})

def parse_model_result(result: Dict[str, Any], keyword: str) -> Dict[str, Any]:
    try:
        data_type = eval(result["data_type"])
    except Exception:
        data_type = str
    return {
        "keyword": keyword,
        "data_description": result["data_description"],
        "data_type": data_type
    }

def generate_dynamic_dict_for_pydantic_model(keyword: str, llm: ChatOpenAI) -> Dict[str, Any]:
    try:
        parser = JsonOutputParser(pydantic_object=DataQualityAnalystModel)
        prompt = create_data_quality_prompt(parser)        
        result = create_chain(keyword, prompt, parser, llm)
        return parse_model_result(result, keyword)
    except Exception:
        return {
            'keyword': keyword,
            'data_description': f"{keyword} present in the text.",
            'data_type': str
        }

def create_pydantic_model_from_dict(pydantic_data: List[Dict[str, Any]], model_name: str = "DynamicModel") -> Any:
    try:
        fields = {
            field["keyword"]: (field["data_type"], Field(default=None, description=field["data_description"]))
            for field in pydantic_data
        }
        return create_model(model_name, **fields)
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    return None

def dynamic_data_extractor(text:str, keywords: List[str]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    pydantic_data = [generate_dynamic_dict_for_pydantic_model(keyword, llm) for keyword in keywords]
    print(pydantic_data)
    dynamic_model = create_pydantic_model_from_dict(pydantic_data)

    if dynamic_model:
        parser = JsonOutputParser(pydantic_object=dynamic_model)
        prompt = PromptTemplate(
            template="Based on the text below answer the question, if not a specific value return null\n{format_instructions}\n{keyword}\n",
            input_variables=["keyword"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        result = create_chain(text, prompt, parser, llm)
        return(result)

def main():

    text = (
    "Acme Corporation, identified by its EIN 12-3456789, has entered into a service agreement with Beta Solutions, LLC. "
    "The contract, valued at $150,000, outlines the terms for consulting services to be provided over a period of one year. "
    "The start date of the contract is October 1, 2024, and the parties have agreed that payments will be made quarterly. "
    "Contact information for Acme Corporation includes their primary address at 123 Main Street, Springfield, and the contact person is John Doe, reachable at john.doe@acme.com."
)

    keywords = [
        "Contract value",
        "EIN (Employer Identification Number)",
        "Company names",
        "Service description",
        "Contract duration",
        "Contract start date"
    ]

    results = dynamic_data_extractor(text, keywords)

    print(results)

if __name__ == "__main__":
    main()