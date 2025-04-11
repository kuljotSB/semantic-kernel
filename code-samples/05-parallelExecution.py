import asyncio
import logging
import sys
import time
from typing import Annotated
import math
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel import Kernel
from dotenv import load_dotenv
from semantic_kernel.planners import SequentialPlanner
import os


def set_up_logging():
    """Set up logging to verify the kernel execute the functions in parallel"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s.%(msecs)03d %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
    )
    # Print only the logs from the chat completion client to reduce the output of the sample
    handler.addFilter(lambda record: record.name == "semantic_kernel.connectors.ai.chat_completion_client_base")

    root_logger.addHandler(handler)
    
    
class Math:
    """
    Description: MathPlugin provides a set of functions to make Math calculations.

    Usage:
        kernel.add_plugin(MathPlugin(), plugin_name="math")

    Examples:
        {{math.Add}} => Returns the sum of input and amount (provided in the KernelArguments)
        {{math.Subtract}} => Returns the difference of input and amount (provided in the KernelArguments)
        {{math.Multiply}} => Returns the multiplication of input and number2 (provided in the KernelArguments)
        {{math.Divide}} => Returns the division of input and number2 (provided in the KernelArguments)
    """

    @kernel_function(
        description="Divide two numbers.",
        name="Divide",
    )
    def divide(
        self,
        number1: Annotated[float, "the first number to divide from"],
        number2: Annotated[float, "the second number to by"],
    ) -> Annotated[float, "The output is a float"]:
        return float(number1) / float(number2)

    @kernel_function(
        description="Multiply two numbers. When increasing by a percentage, don't forget to add 1 to the percentage.",
        name="Multiply",
    )
    def multiply(
        self,
        number1: Annotated[float, "the first number to multiply"],
        number2: Annotated[float, "the second number to multiply"],
    ) -> Annotated[float, "The output is a float"]:
        return float(number1) * float(number2)

    @kernel_function(
        description="Takes the square root of a number",
        name="Sqrt",
    )
    def square_root(
        self,
        number1: Annotated[float, "the number to take the square root of"],
    ) -> Annotated[float, "The output is a float"]:
        return math.sqrt(float(number1))

    @kernel_function(name="Add")
    def add(
        self,
        number1: Annotated[float, "the first number to add"],
        number2: Annotated[float, "the second number to add"],
    ) -> Annotated[float, "the output is a float"]:
        return float(number1) + float(number2)

    @kernel_function(
        description="Subtracts value to a value",
        name="Subtract",
    )
    def subtract(
        self,
        number1: Annotated[float, "the first number"],
        number2: Annotated[float, "the number to subtract"],
    ) -> Annotated[float, "the output is a float"]:
        return float(number1) - float(number2)



async def parallel_execution():
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
    
    kernel.add_plugin(Math(), "MathPlugin")

    plugin = kernel.add_plugin(parent_directory="../plugins/prompt_templates/", plugin_name="basic_plugin")

    contact_function = plugin["greeting"]
    
    #query which will execute in parallel
    query1 = "greet kuljot who is of age 19 and tell me how much is 10 divided by 2"
    
    
    
    
    arguments = KernelArguments(
        settings=PromptExecutionSettings(
            # Set the function_choice_behavior to auto to let the model
            # decide which function to use, and let the kernel automatically
            # execute the functions.
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
    )
    
    
    
    result = await kernel.invoke_prompt(query1, arguments=arguments)
    print(result)

   
    
    
async def sequential_execution():
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
    
    kernel.add_plugin(Math(), "MathPlugin")

    plugin = kernel.add_plugin(parent_directory="../plugins/prompt_templates/", plugin_name="basic_plugin")

    contact_function = plugin["greeting"]
    
    #query which will execute in sequence
    query1 = "greet kuljot who is of age 19 and tell me how much is 10 divided by 2"
    
    planner = SequentialPlanner(
    kernel,
    service_id
    )
    
    sequential_plan = await planner.create_plan(query1)
    
    print("The plan's steps are:")
    for step in sequential_plan._steps:
        print(
            f"- {step.description.replace('.', '') if step.description else 'No description'} using {step.metadata.fully_qualified_name} with parameters: {step.parameters}"
        )
    
    result = await sequential_plan.invoke(kernel)
    
    print(result)
    
   
if __name__ == "__main__":
    set_up_logging()

    asyncio.run(parallel_execution())