from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
import os
import asyncio
import time
from dotenv import load_dotenv

kernel = Kernel()

load_dotenv()

service_id = "default"
kernel.add_service(
    AzureChatCompletion(service_id=service_id,
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
                        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
)

plugin = kernel.add_plugin(parent_directory = "../plugins/prompt_templates/", plugin_name = "basic_plugin")

greeting_function = plugin["greeting"]

async def greeting():
    return await kernel.invoke(greeting_function, KernelArguments(name="kuljot", age="18"))

greeting_response =  asyncio.run(greeting())

print(greeting_response)