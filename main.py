__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

app = FastAPI()


# Configuração do modelo de linguagem e agentes
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_llm = ChatOpenAI(model_name='gpt-4-1106-preview', api_key=openai_api_key)

agente_pesquisa_venda = Agent(
    role='Agente de Vendas do WhatsApp com a missão de gerar vendas',
    goal='Identificar se o cliente deseja comprar e se manter na conversa ao máximo',
    backstory='Você é um agente que trabalha em uma empresa como assistente no WhatsApp, ajudando os leads a completarem suas compras.',
    llm=openai_llm
)

agente_suporte_sdr = Agent(
    role='Agente de Vendas do WhatsApp com a missão de gerar vendas',
    goal='Identificar se o cliente deseja comprar e se manter na conversa ao máximo',
    backstory='Você é um agente que trabalha em uma empresa como assistente no WhatsApp, ajudando os leads a completarem suas compras.',
    llm=openai_llm
)


@app.post("/execute_task")
async def execute_task(task_type: str, data: dict):
    """
    Recebe o tipo da tarefa e os dados para realizar a tarefa de agente.
    """

    # Identificar o tipo de tarefa e criar a instância correta
    if task_type == "pesquisa_venda":
        task = Task(
            description=data.get("description", "Descrição de tarefa de pesquisa de venda"),
            agent=agente_pesquisa_venda
        )
    elif task_type == "suporte_sdr":
        task = Task(
            description=data.get("description", "Descrição de tarefa de suporte SDR"),
            agent=agente_suporte_sdr
        )
    else:
        raise HTTPException(status_code=400, detail="Tipo de tarefa inválido")

    # Executa a tarefa e coleta o resultado
    try:
        crew = Crew(agents=[agente_pesquisa_venda, agente_suporte_sdr], tasks=[task], verbose=True)
        result = crew.kickoff()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result}