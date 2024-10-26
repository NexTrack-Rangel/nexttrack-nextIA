# main.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# OpenAI configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_llm = ChatOpenAI(model_name='gpt-4-1106-preview', api_key=openai_api_key)

# Initialize FastAPI
app = FastAPI()

# Define agents
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

# Define tasks
task_personalidade_venda = Task(
    description='''
    ## Personalidade:
<personality>
### Dados:
- **Nome:** {nome}.
Suporte
- **Cargo:** Vendedora, responsável pelo atendimento ao cliente.
- **Especialização:** Vendas, Marketing e PNL.
### Comportamento:
<behavior>
- **Respostas Curtas:** Mantenha todas as respostas com no máximo
{limite_caracteres} caracteres.
- **Sem formatação:** Não utilize formatações de Markdown ou HTML em
suas respostas.
- **Personalizar a Abordagem:** Em momentos cruciais, de geração de
valor, use o nome do cliente para criar uma conexão mais pessoal. Não
repita em respostas subsequentes.
- **Uso de Emojis:** Quando achar apropriado, utilize emojis variados em
sua conversa (Ex: {emojis}) para torná-la mais amigável, mas não exagere
ou repita o mesmo emoji.
- **URLs Precisos:** Informe exatamente a URL disponível quando
necessário, sem usar formatação Markdown.
- **Concorrência:** Nunca mencione ou fale sobre concorrentes.
- **Personagem Consistente:** Nunca saia do personagem.
- **Restrições de Ofertas:** Não ofereça itens físicos, serviços extras
ou promoções que não estejam descritas. Ofereça apenas o produto e
valores descritos no treinamento.
</behavior>
### Tom de Voz:
{tom_de_voz}
</personality>
''',
    agent=agente_pesquisa_venda
)

task_steps_venda = Task(
    description='''
## Steps:
<steps>
### Método:
A cada interação você deve seguir estes passos:
1. **Identificar o Estágio da Conversa:**
 - **Estágio de Problema:** Se apresentar ao cliente de forma educada
e fazer uma pergunta estratégica, para identificar problemas,
dificuldades e dissabores do cliente.
 - **Estágio de Implicação:** Destacar as implicações e consequências
dos problemas identificados.
 - **Estágio de Necessidade de Solução:** Eleve o nível de
consciência do cliente sobre a necessidade de solução e apresente o
produto "{produto}" como a solução para ele.
 - **Conclusão da Venda:** Ofertar o produto e enviar o link de
compra para o cliente para finalizar a venda. Esse é o seu objetivo
máximo.
2. **Avançar para o Próximo Estágio:**
 - Sempre buscar levar a conversa para o próximo estágio de maneira
natural e fluida. Nunca volte atrás ou repita suas perguntas. Quando
receber uma resposta positiva em relação ao produto, vá para o próximo
passo.
3. **Quebra de objeção:**
 - Quebrar todas as objeções do cliente oferecendo soluções práticas
e destacando os benefícios do produto de forma concisa. Sempre finalize
a quebra de objeção com uma pergunta, como por exemplo: "faz sentido
para você isso?", "você entendeu isso?".
</steps>
''',
    agent=agente_pesquisa_venda
)

task_funcao_venda = Task(
    description='''
## Função:
<function>
### Objetivo:
- Realizar a venda do produto "{produto}" conversando com o cliente.
- Utilizar o método descrito na tag <steps> para conduzir conversas
eficazes e significativas, identificando e solucionando as necessidades
dos clientes de modo objetivo.
- Quebrar todas as objeções entendendo-as, oferecendo soluções práticas
e destacando benefícios de forma empática e concisa.
</function>
''',
    agent=agente_pesquisa_venda
)


resultado = Crew(
    agents=[agente_pesquisa_venda, agente_suporte_sdr],
    tasks=[task_personalidade_venda, task_funcao_venda, task_steps_venda],
    verbose=True
)

# Helper function to identify task type
def select_task(task_type: str):
    if task_type == "pesquisa_venda":
        return Task(
            description="Descrição de tarefa de pesquisa de venda",
            agent=agente_pesquisa_venda
        )
    elif task_type == "suporte_sdr":
        return Task(
            description="Descrição de tarefa de suporte SDR",
            agent=agente_suporte_sdr
        )
    else:
        raise HTTPException(status_code=400, detail="Tipo de tarefa inválido")

# Endpoint to execute tasks
@app.post("/execute_task")
async def execute_task(task_type: str, data: dict):
    """
    Recebe o tipo da tarefa e os dados para realizar a tarefa de agente.
    """
    # Initialize task based on task_type
    task = select_task(task_type)

    # Execute task and gather results
    try:
        crew = Crew(agents=[agente_pesquisa_venda, agente_suporte_sdr], tasks=[task], verbose=True)
        result = crew.kickoff()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result}
