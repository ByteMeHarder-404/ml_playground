from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal,List
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv(
)

model=HuggingFaceEndpoint(repo_id='google/gemma-3-4b-it',task='text-generation')

class Exam(BaseModel):
    ques: List[str] =Field(description='Write 5 questions that you can make from the paragraph.')
    summ:str=Field(description='Give the key points of the input paragraph')
prs=PydanticOutputParser(pydantic_object=Exam)
tem=PromptTemplate(template='Generate 5 questions and key points from {inp} \n{form}',
                   input_variables=['inp'],
                   partial_variables={'form':prs.get_format_instructions()})
strt=tem|model|prs
res=strt.invoke({'inp':'''Current Vision-Language-Action (VLA) models rely on fixed computational depth, expending the same amount of compute on simple adjustments and complex multi-step manipulation. While Chain-of-Thought (CoT) prompting enables variable computation, it scales memory linearly and is ill-suited for continuous action spaces. We introduce Recurrent-Depth VLA (RD-VLA), an architecture that achieves computational adaptivity via latent iterative refinement rather than explicit token generation. RD-VLA employs a recurrent, weight-tied action head that supports arbitrary inference depth with a constant memory footprint. The model is trained using truncated backpropagation through time (TBPTT) to efficiently supervise the refinement process. At inference, RD-VLA dynamically allocates compute using an adaptive stopping criterion based on latent convergence. Experiments on challenging manipulation tasks show that recurrent depth is critical: tasks that fail entirely (0 percent success) with single-iteration inference exceed 90 percent success with four iterations, while simpler tasks saturate rapidly. RD-VLA provides a scalable path to test-time compute in robotics, replacing token-based reasoning with latent reasoning to achieve constant memory usage and up to 80x inference speedup over prior reasoning-based VLA models'''})
print(res)