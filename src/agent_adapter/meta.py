from typing import Protocol, TypeVar

import dotenv
from typing_extensions import runtime_checkable

T = TypeVar("T")

dotenv.load_dotenv()


@runtime_checkable
class AgentAdapter(Protocol[T]):
    async def produce(self, template: str, output_type: type[T], **kwargs) -> T:
        """
        Adapter protocol for agents to produce outputs based on a given template and output type.
        Args:
            template (str): The input template or prompt for the agent.
            output_type (type[T]): The expected output type.
            **kwargs: Additional keyword arguments for the agent.
        Returns:
            T: The output produced by the agent, matching the specified output type.
        """
        pass
