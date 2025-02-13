from semantic_kernel import Kernel
import os
import asyncio
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
from semantic_kernel.planners import SequentialPlanner
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import BingGroundingTool

load_dotenv()
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment_name = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL")
ai_project_connection_string = os.getenv("AI_PROJECT_CONNECTION_STRING")
bing_connection_name = os.getenv("BING_CONNECTION_NAME")

project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=ai_project_connection_string
        )

class Agents:
    @kernel_function(
        description="This function will be used to use an azure ai agent with web grounding capability using Bing Search API",
        name="WebSearchAgent"
    )
    def web_search_agent(
        self,
        query: Annotated[str, "The user query for which the contextual information needs to be fetched from the web"]
        
    ) -> Annotated[str, "The response from the web search agent"]:
        bing_connection = project_client.connections.get(connection_name=bing_connection_name)
        conn_id = bing_connection.id
        
        bing = BingGroundingTool(connection_id=conn_id)
        
        final_response: str = ""
        
        agent = project_client.agents.create_agent(
        model=azure_openai_deployment_name,
            name="bing-assistant",
            instructions="You are a helpful assistant",
            tools=bing.definitions,
            headers={"x-ms-enable-preview": "true"},
        )
        thread = project_client.agents.create_thread()
            
        message = project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=query,
            )
            
        run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
            
        messages = project_client.agents.list_messages(thread_id=thread.id)
            
        print(messages.data[0].content[0].text.value)
            
        final_response = messages.data[0].content[0].text.value
            
            
        return final_response
    
    @kernel_function(
       description="This function will use an azure ai agent to prepare a script for a news reporter based on latest information for a specific topic",
         name="NewsReporterAgent"
   )
    def news_reporter_agent(
        self,
        topic: Annotated[str, "The topic for which the latest information/news has been fetched"],
        latest_news: Annotated[str,"The latest information for a specific topic"]
    ) -> Annotated[str, "the response from the NewsReporterAgent which is the script for a news reporter"]:
        final_response: str = ""
        
        agent = project_client.agents.create_agent(
        model=azure_openai_deployment_name,
        name="news-reporter",
        instructions="""You are a helpful assistant that is meant to prepare a script for a news reporter based on the latest information for a specific topic both of which you will be given.
            The news channel is named MSinghTV and the news reporter is named John. You will be given the topic and the latest information for that topic. Prepare a script for the news reporter John based on the latest information for the topic.""",
            headers={"x-ms-enable-preview": "true"},
        )
        thread = project_client.agents.create_thread()
            
        message = project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=f"""The topic is {topic} and the latest information is {latest_news}""",
            )
            
        run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
            
        messages = project_client.agents.list_messages(thread_id=thread.id)
            
        print(messages.data[0].content[0].text.value)
            
        final_response = messages.data[0].content[0].text.value
            
            
        return final_response
            
        
    
    
    
        

kernel = Kernel()

service_id = "default"


kernel.add_service(
    AzureChatCompletion(service_id=service_id,
                        api_key=azure_openai_key,
                        deployment_name=azure_openai_deployment_name,
                        endpoint = azure_openai_endpoint
    )
)


planner = SequentialPlanner(
    kernel,
    service_id
)

agents_plugin = kernel.add_plugin(Agents(), "Agents")

goal = f"prepare a news script for John on latest news for India?"
        
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




