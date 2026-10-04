import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dotenv import load_dotenv

from agno.agent import Agent
from agno.media import Image
from common import make_model
from agno.models.openrouter import OpenRouter
from agno.models.openai import OpenAIResponses

# text_model = OpenRouter(id=os.getenv("TEXT_MODEL_ID"))
# vision_model = OpenRouter(id=os.getenv("VISION_MODEL_ID"))


load_dotenv(ROOT / ".env")

VISION_MODEL_ID = os.getenv(
    "VISION_MODEL_ID",
    "inclusionai/ling-3.0-flash-vl:free"
)
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
)


agent = Agent(
    # model=make_model(),

    # model=OpenRouter(id="openrouter/free"),
    # model=OpenRouter(id=VISION_MODEL_ID),
    model=OpenAIResponses(id=VISION_MODEL_ID),
    instructions=[
        "Describe only what can reasonably be inferred from the image.",
        "Separate direct observations from hypotheses.",
    ],
    markdown=True,
)

if __name__ == "__main__":
    default_image = Path(__file__).parent / "sample.png"
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_image

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Usage:")
        print("  uv run python 09_multimodal/image_agent.py path/to/photo.png")
        raise SystemExit(1)

    agent.print_response(
        "Describe this image. Identify important objects and suggest what the scene "
        "might be showing. Clearly label any uncertainty.",
        images=[Image(filepath=str(image_path))],
        stream=True,
    )
