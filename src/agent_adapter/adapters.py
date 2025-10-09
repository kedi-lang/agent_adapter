import dspy
from dspy import LM
from pydantic_ai import Agent
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.output import OutputDataT

from .meta import T


class PydanticAdapter(Agent[AgentDepsT, OutputDataT]):
    async def produce(
        self, template: str, output_type: OutputDataT, **kwargs
    ) -> OutputDataT:
        """
        Adapter to use Pydantic AI agents.
        """
        return (await self.run(template, output_type=output_type, **kwargs)).output


class DSPyAdapter:
    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        temperature: float = 1.0,
        provider: dspy.Provider | None = None,
        **kwargs,
    ):
        dspy.settings.configure(
            lm=LM(
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        )

    async def produce(
        self,
        template: str,
        output_type: type[T] | str | dspy.Signature | None = None,
        **kwargs,
    ) -> T:
        """
        Adapter to use DSPy agents.
        """
        assert issubclass(output_type, dspy.Signature) or type(str) is type, ValueError(
            "Signature or type must be passed."
        )
        kwargs.pop("user_prompt", None)
        instructions = kwargs.pop("instructions", None)
        kw = {} if not instructions else {"instructions": instructions}
        if issubclass(output_type, dspy.Signature):
            prediction = dspy.Predict(output_type)
            return prediction(**kwargs, user_prompt=template)
        else:
            _type_repr = repr(output_type)
            _type_repr = (
                _type_repr
                if not _type_repr.startswith("<class")
                else output_type.__name__
            )
            sig = f"user_prompt: str -> output: {_type_repr}"
            prediction = dspy.Predict(dspy.Signature(sig, **kw))
            return getattr(prediction(user_prompt=template), "output", None)
