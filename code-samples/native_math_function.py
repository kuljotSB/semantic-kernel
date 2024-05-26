import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
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

math_plugin = kernel.import_native_plugin_from_directory("../plugins/native_plugins/", "mathPlugin")

sqrt_function = math_plugin["Sqrt"]

async def square_root():
    return await kernel.invoke(sqrt_function, number1=4 )

print(asyncio.run(square_root()))