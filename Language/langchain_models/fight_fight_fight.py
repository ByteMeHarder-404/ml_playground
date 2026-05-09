import os, json, re
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
class Arg(BaseModel):
    argument: str = Field(description="The core thesis of the position.")
    evidence: List[str] = Field(description="List of factual or logical proofs.")
    assumptions: List[str] = Field(description="Implicit beliefs taken for granted.")
    confidence: float = Field(description="Soundness score from 0.0 to 1.0.")

class Jud(BaseModel):
    strongest_pro: str = Field(description="Pro's strongest point.")
    strongest_con: str = Field(description="Con's strongest point.")
    synthesis: str = Field(description="Logical middle ground.")
    disagreement: float = Field(description="Disagreement score from 0.0 to 1.0.")

class DEb(BaseModel):
    claim: str
    pro_analysis: Arg
    con_analysis: Arg
    judge_report: Jud
    summary: str
    uncertainty: Optional[str] = None
arg_parser = JsonOutputParser(pydantic_object=Arg)
jud_parser = JsonOutputParser(pydantic_object=Jud)

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    temperature=0.1,
    max_new_tokens=2048
)
model = ChatHuggingFace(llm=llm)
def clean_json_string(content: str) -> str:
    content = re.sub(r"```json\s?|\s?```", "", content).strip()
    content = content.replace('\\"', "'") 
    return content
system_base = (
    "You are a professional analyst. You MUST output ONLY a JSON object. "
    "Use the exact keys provided in the schema instructions. "
    "If you want to use quotes inside a sentence, use single quotes (').\n"
    "{format_instructions}"
)

prmpro = ChatPromptTemplate.from_messages([
    ("system", system_base),
    ("human", "Argue in FAVOR of: {claim}")
]).partial(format_instructions=arg_parser.get_format_instructions())

prmcon = ChatPromptTemplate.from_messages([
    ("system", system_base),
    ("human", "Argue AGAINST: {claim}")
]).partial(format_instructions=arg_parser.get_format_instructions())

prmjud = ChatPromptTemplate.from_messages([
    ("system", system_base),
    ("human", "Claim: {claim}\n\nPro: {pro}\n\nCon: {con}")
]).partial(format_instructions=jud_parser.get_format_instructions())

prmref = ChatPromptTemplate.from_messages([
    ("system", system_base),
    ("human", "Refine your position. Address this point: {opp_point}\n\nCurrent: {current_pos}")
]).partial(format_instructions=arg_parser.get_format_instructions())
def fight(claim: str, rnd=2, thrs=0.7):
    base_chain = model | RunnableLambda(lambda msg: clean_json_string(msg.content))
    
    print("--- Round 1: Initial Stances ---")
    pro_dict = (prmpro | base_chain | arg_parser).invoke({'claim': claim})
    proa = Arg(**pro_dict)
    con_dict = (prmcon | base_chain | arg_parser).invoke({'claim': claim})
    cona = Arg(**con_dict) 
    jud_dict = (prmjud | base_chain | jud_parser).invoke({
        'claim': claim, 'pro': proa.model_dump_json(), 'con': cona.model_dump_json()
    })
    judy = Jud(**jud_dict)
    
    crnd = 1
    while judy.disagreement > thrs and crnd < rnd:
        crnd += 1
        print(f"--- Round {crnd}: Disagreement {judy.disagreement} ---")
        pro_dict = (prmref | base_chain | arg_parser).invoke({
            'opp_point': judy.strongest_con, 'current_pos': proa.model_dump_json()
        })
        proa = Arg(**pro_dict)
        con_dict = (prmref | base_chain | arg_parser).invoke({
            'opp_point': judy.strongest_pro, 'current_pos': cona.model_dump_json()
        })
        cona = Arg(**con_dict)
        jud_dict = (prmjud | base_chain | jud_parser).invoke({
            'claim': claim, 'pro': proa.model_dump_json(), 'con': cona.model_dump_json()
        })
        judy = Jud(**jud_dict)

    return DEb(
        claim=claim,
        pro_analysis=proa,
        con_analysis=cona,
        judge_report=judy,
        summary=judy.synthesis
    )

res = fight("AI will replace most jobs")
print("\n" + "="*50)
print(res.model_dump_json(indent=2))