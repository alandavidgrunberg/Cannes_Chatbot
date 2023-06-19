import gradio as gr
import pandas as pd
import time

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent, Tool, ZeroShotAgent, AgentExecutor
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import TokenTextSplitter

### CREATING DATAFRAME AGENT:

df = pd.read_csv('data/complete_data_one_hot.csv')
# ^dataframe of all movies
# English title, Original title, Director(s), Production countrie(s), + 11 screening categories (one hot encoded)

with open('data/df_agent_prefix.txt', 'r') as file:
    df_agent_prefix = file.read()
# ^prefix is prompt that is fed to the bot prepending user's question every time agent used. See text file for content

df_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, prefix=df_agent_prefix, verbose=True)
# ^create agent (tool for the bot to use) which can read dataframes in a virtual python repl


### CREATING TEXT VECTORSTORES:

wiki_film_loader = DirectoryLoader("data/film_summaries/from_wikipedia", glob="*.txt")
# # ^loading movie summaries (pre-scraped from wikipedia)
search_film_loader = DirectoryLoader("data/film_summaries/from_search", glob="*.txt")
 # ^loading more movie summaries (pre-scraped from google search top result)

festival_info_loader = DirectoryLoader("data/festival_info", glob="*.txt")
 # ^loading festival info (pre-scraped from google search top result)

film_summaries_index = VectorstoreIndexCreator(text_splitter=TokenTextSplitter(chunk_size=500, chunk_overlap=20)).from_loaders([wiki_film_loader, search_film_loader])
# # ^creating vector index of movie summaries

festival_info_index = VectorstoreIndexCreator(text_splitter=TokenTextSplitter(chunk_size=200, chunk_overlap=20)).from_loaders([festival_info_loader])
# ^creating vector index of movie summaries



### PUTTING TOOLBOX TOGETHER:

tools = []

tools.append(
    Tool(
        name="python_repl_ast",
        func=df_agent.run,
        description="Useful when you need to count movies, directors, countries, etc. at the upcoming Cannes Film Festival. Useful when asked 'How many' Do not use for finding film genres. Do not use for questions about juries or the red carpet.",
        verbose = True # change to false to not show agent 'thinking' through its actions, and just output final answer
        )
)

tools.append(
    Tool(
        name="film summaries",
        func=film_summaries_index.query,
        description="Useful when you are asked about the plot of a film at the upcoming Cannes Film Festival, the actors in the film, and the film's genre. Use for finding film genres. Do not use for questions about juries or the red carpet=.",
        verbose = True # change to false to not show agent 'thinking' through its actions, and just output final answer
        )
)

tools.append(
    Tool(
        name="festival general info",
        func=festival_info_index.query,
        description="Useful when you are asked for general info about the upcoming Cannes Film Festival, such as: When it will take place? Who will judge the films? Who is on the jury? Who was on the red carpet?",
        verbose = True # change to false to not show agent 'thinking' through its actions, and just output final answer
        )
)
# ^bot will pick which tool to use depending on the question asked and the tool description

### BUILDING MEMORY CHAIN

prefix = """Have a conversation with a human, answering the following questions about the upcoming Cannes Film Festival as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)

### CREATING MASTER AGENT CHAIN WITH MEMORY AND ACCESS TO TOOLBOX

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
# ^agentchain ready for queries

### CONNECTING TO GRADIO FRONTEND

spacing = "<br>"
header_content = "<p style='text-align: center;''>Hello there! I am a conversation bot trained on Cannes 2023 data a few weeks before the festival. I was designed to help cinephiles learn more before the big event. Ask me about the festival as if it hasn’t happened yet and you’d like to learn more. I’ll be happy to answer your questions.</p>"
footer_content = "<p style='text-align: center;''>Check out my <a href='https://github.com/alandavidgrunberg/Cannes_Chatbot'>GitHub Repo</a> to learn how I was created.</p>"

with gr.Blocks(title="Cannes 2023 Q&A", theme="gradio/monochrome") as demo:
    spacer = gr.Markdown(spacing)
    header = gr.Markdown(header_content)
    chatbot = gr.Chatbot(label = 'Cannes Bot')
    textbox = gr.Textbox(label = 'Input:', value = 'Tell me about the upcoming festival!')
    button = gr.Button("Submit")
    clear = gr.ClearButton([textbox, chatbot])
    footer = gr.Markdown(footer_content)
    spacer = gr.Markdown(spacing)

    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        bot_message = agent_chain.run(f"Answer the following question using the tools provided. Do not make up the answer if you can't find it using the tools. Always talk about the festival in the future tense, it hasn't happened yet. Question: {history[-1][0]}")
                                        # where the magic happens (connecting model)
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.02)
            yield history

    response = textbox.submit(user, inputs=[textbox, chatbot], outputs=[textbox, chatbot], queue=False).then(
        bot, inputs=chatbot, outputs=chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [textbox], queue=False)

    response = button.click(user, inputs=[textbox, chatbot], outputs=[textbox, chatbot], queue=False).then(
        bot, inputs=chatbot, outputs=chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [textbox], queue=False)

demo.queue()
demo.launch()
