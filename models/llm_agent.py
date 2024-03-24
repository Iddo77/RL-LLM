from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

from utils import convert_image_to_base64


# and environment variable OPENAI_API_KEY must be set with the OpenAI key


def query_image_with_text(image, text):

    base64_image = convert_image_to_base64(image)
    chat = ChatOpenAI(model='gpt-4-vision-preview', max_tokens=512)
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ]
            )
        ]
    )

    return response

