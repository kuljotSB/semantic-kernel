from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
import os
from dotenv import load_dotenv
import asyncio
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

plugin = kernel.add_plugin(parent_directory="../plugins/prompt_templates/", plugin_name="basic_plugin")

contact_function = plugin["contact_information"]

async def contact():
    return await kernel.invoke(contact_function, KernelArguments(name="kuljot", contact_number="1234567890", email_id="hello@gmail.com", address="1234, 5th Avenue, New York, NY 10001"))

print(asyncio.run(contact()))
