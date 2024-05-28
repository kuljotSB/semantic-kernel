from semantic_kernel import Kernel
import os
import asyncio
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
from semantic_kernel.planners import SequentialPlanner
kernel = Kernel()

service_id = "default"

load_dotenv()
kernel.add_service(
    AzureChatCompletion(service_id=service_id,
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
                        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
)

planner = SequentialPlanner(
    kernel,
    service_id
)

kernel.add_plugin(parent_directory="../plugins/prompt_templates/", plugin_name="writerPlugin")

for plugin_name, plugin in kernel.plugins.items():
    for function_name, function in plugin.functions.items():
        print(f"Plugin: {plugin_name}, Function: {function_name}")
        
text=" "
        
with open("../data/chatgpt.txt", "r") as file:
    chatgpt = file.read()
    text = text+chatgpt
    
    
goal = f"summarise this text: {text} and email it to sam@gmail.com "
        
async def call_planner():
    return await planner.create_plan(goal)

sequential_plan = asyncio.run(call_planner())

print("The plan's steps are:")
for step in sequential_plan._steps:
    print(
        f"- {step.description.replace('.', '') if step.description else 'No description'} using {step.metadata.fully_qualified_name} with parameters: {step.parameters}"
    )

async def generate_answer():
    return await sequential_plan.invoke(kernel)

result = asyncio.run(generate_answer())

print(result)
