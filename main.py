import openai
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from typing import List, Optional

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure templates directory exists
os.makedirs("templates", exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output": None, "purpose": "", "examples": [{"input": "", "output": ""}]})

@app.post("/generate", response_class=HTMLResponse)
async def generate_prompt(request: Request, 
                    purpose: str = Form(...),
                    example_input: List[str] = Form(...),
                    example_output: List[str] = Form(...)):
    examples = [{"input": i, "output": o} for i, o in zip(example_input, example_output)]
    prompt = await build_system_prompt(purpose, examples)
    return templates.TemplateResponse("index.html", {"request": request, "output": prompt, "purpose": purpose, "examples": examples})

@app.get("/example/email", response_class=HTMLResponse)
def email_example(request: Request):
    purpose = "Classify emails as 'spam' or 'not spam'. Output 'spam' for unwanted emails and 'not spam' for legitimate emails."
    examples = [
        {"input": "Congratulations! You've won a $1000 gift card. Click here to claim.", "output": "spam"},
        {"input": "Dear team, please find attached the minutes from today's meeting.", "output": "not spam"},
        {"input": "Limited time offer! Buy now and save 50%.", "output": "spam"},
        {"input": "Your Amazon order has shipped.", "output": "not spam"}
    ]
    return templates.TemplateResponse("index.html", {"request": request, "output": None, "purpose": purpose, "examples": examples})

async def build_system_prompt(purpose: str, examples: List[dict]) -> str:
    example_text = '\n'.join([
        f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples if ex['input'] or ex['output']
    ])
    system_prompt = """You are an expert prompt engineer. Generate a clear, concise system prompt for an AI assistant that performs multiclass classification.
"""

    user_prompt = f"""Purpose: {purpose}
Examples:
{example_text}
The system prompt should instruct the AI to classify inputs into one of the predefined classes, use the examples as guidance, and respond only with the class label."""
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=system_prompt,
            input=user_prompt
        )

        return response.output_text
    except Exception as e:
        return f"[OpenAI API Error] {e}" 