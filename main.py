from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI

# Carregar variáveis de ambiente
load_dotenv()

# Obter a chave de API do OpenAI do arquivo .env
openai_api_key = os.getenv('OPENAI_API_KEY')

# Inicializar o modelo LLM da OpenAI com a chave de API e o modelo apropriado
openai_llm = ChatOpenAI(model_name='gpt-4-1106-preview', api_key=openai_api_key)

# Definir o primeiro agente (Agente de Vendas)
agente_pesquisa_venda = Agent( 
    role='Agente de Vendas do WhatsApp com a missão de gerar vendas',
    goal='Identificar se o cliente deseja comprar e se manter na conversa ao máximo',
    backstory='Você é um agente que trabalha em uma empresa como assistente no WhatsApp, ajudando os leads a completarem suas compras.',
    llm=openai_llm
)

# Definir o segundo agente (Suporte SDR)
agente_suporte_sdr = Agent(
    role='Suporte de Whatsapp que tem como missão gerar vendas',
    goal='Identificar se o cliente deseja comprar',
    backstory='Você é um agente que trabalha em uma empresa como assistente de WhatsApp dando suporte SDR e ajuda os leads a completarem suas compras.',
    llm=openai_llm
)

# Definir a primeira tarefa (Personalidade e Comportamento)
tarefa_pesquisa_venda1 = Task(
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
- **Respostas Curtas:** Mantenha todas as respostas com no máximo {limite_caracteres} caracteres.
- **Sem formatação:** Não utilize formatações de Markdown ou HTML em suas respostas.
- **Personalizar a Abordagem:** Em momentos cruciais, use o nome do cliente para criar uma conexão mais pessoal.
- **Uso de Emojis:** Utilize emojis de forma moderada e adequada, sem exagero.
- **URLs Precisos:** Informe a URL exata, sem formatação Markdown.
- **Concorrência:** Nunca mencione concorrentes.
- **Personagem Consistente:** Nunca saia do personagem.
- **Restrições de Ofertas:** Ofereça apenas o produto descrito no treinamento.
</behavior>
### Tom de Voz:
{tom_de_voz}
</personality>
    ''',
    agent=agente_pesquisa_venda
)

# Definir a segunda tarefa (Função e Objetivo)
tarefa_pesquisa_funcion2 = Task(
    description='''

    ## Função:
<function>
### Objetivo:
- Realizar a venda do produto "{produto}" conversando com o cliente.
- Utilizar o método descrito na tag <steps> para conduzir conversas eficazes.
- Quebrar objeções e destacar benefícios de forma empática e concisa.
</function>
    ''',
    agent=agente_pesquisa_venda
)

# Definir a terceira tarefa (Steps do Processo de Vendas)
tarefa_pesquisa_steps3 = Task(
    description='''

    ## Steps:
<steps>
### Método:
A cada interação, siga estes passos:
1. **Identificar o Estágio da Conversa:**
   - Estágio de Problema, Implicação e Solução.
2. **Avançar para o Próximo Estágio:** 
   - Mover a conversa naturalmente sem repetir perguntas.
3. **Quebra de objeção:** 
   - Quebrar objeções e destacar benefícios de forma clara e concisa.
</steps>
    ''',
    agent=agente_pesquisa_venda
)

# Criar a equipe (Crew) e associar agentes e tarefas
equipe = Crew(
    agents=[agente_pesquisa_venda, agente_suporte_sdr],
    tasks=[tarefa_pesquisa_venda1, tarefa_pesquisa_funcion2, tarefa_pesquisa_steps3],
    verbose=True
)

# Executar a equipe (Crew)
resultado = equipe.kickoff()

# Exibir o resultado
print(resultado)
