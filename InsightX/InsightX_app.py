import os
import io
import requests
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import arxiv
import PyPDF2
from datetime import datetime 



from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Reasoning-lab")

if "results_arxiv" not in st.session_state:
    st.session_state.results_arxiv = []
if "pdf_analysis" not in st.session_state:
    st.session_state.pdf_analysis = ""
if "experiment" not in st.session_state:
    st.session_state.experiment = ""
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 6000  # Default value
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role":'system', "content":'You are a helpfull assistant'}]

# Model configuration
gpt_4o_mini = OpenAI(model="gpt-4o-mini", api_key=os.environ['AIML_API_KEY'], api_base="https://api.aimlapi.com")
gpt_o1_mini = OpenAI(model="o1-mini", api_key=os.environ['AIML_API_KEY'], api_base="https://api.aimlapi.com", max_tokens=st.session_state.max_tokens)

st.markdown(
    f"""<style>div[data-testid="stExpander"] div[role="button"] p {{
        font-size: 10rem;
    }}</style>""",
    unsafe_allow_html=True
)

#############
# Functions #
#############

def search_arxiv(query, max_results=5, option_sort_by='Relevance'):
    sort_by_dict = {
        'Relevance': arxiv.SortCriterion.Relevance,
        'Last Updated Date': arxiv.SortCriterion.LastUpdatedDate,
        'Submitted Date': arxiv.SortCriterion.SubmittedDate,
    }
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by_dict.get(option_sort_by, arxiv.SortCriterion.Relevance),
        
    )
    results = []
    for result in search.results():
        print(result.categories)
        results.append({
            'title': result.title,
            'summary': result.summary,
            # 'categories': ', '.join(cat for cat in result.categories),
            'published': result.published.strftime("%Y-%m-%d"),
            'updated': result.updated.strftime("%Y-%m-%d"),
            'authors': ', '.join(author.name for author in result.authors),
            'url': result.pdf_url
        })
    return results

def analyze_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Use gpt-4o-mini for simple analysis
    analysis = gpt_4o_mini.complete(f"Analyze the following text and provide a summary of key findings: {text[:4000]}")  # Limit to 4000 characters for simplicity
    st.session_state.pdf_analysis = analysis.text
    return analysis.text

def create_hypothetical_experiment(topic, max_tokens):
    # Use gpt-o1-mini for complex tasks
    experiment = gpt_o1_mini.complete(f"Create a hypothetical experiment on the following topic: {topic}. Include hypothesis, methodology, and possible outcomes.", max_tokens=max_tokens)
    st.session_state.experiment = experiment.text
    return experiment.text

def analyze_chat_history(messages):
    # Basic analysis to determine which model to use

    messages_temp = messages[1:]
    messages_temp.append(
        ChatMessage(
            role="system",
            content="Based on the user's last message, determine whether to use 'gpt-4o-mini' for general responses or 'o1-mini' for complex reasoning tasks. Please respond with only 'gpt-4o-mini' or 'o1-mini'."
        )
    )

    response = gpt_4o_mini.chat(messages_temp)
    # print(f"{response}")
    print(response)
    if "o1-mini" in str(response).lower():
        return "o1-mini"

    return "gpt-4o-mini"

def update_o1_model_tokens():
    max_tokens = st.session_state.exp_max_tokens
    gpt_o1_mini._update_max_tokens(max_tokens=max_tokens)

def main():
    st.title("Reasoning-lab: Explore, experiment and find scientific solutions")

    menu = ["Search Articles", "PDF Analysis", "Hypothetical Experiments", "Reasoning Chat"]
    choice = st.sidebar.selectbox("Select a function", menu)

    with st.sidebar:
        max_tokens = st.number_input("Max tokens for **o1-mini** model", key='exp_max_tokens', min_value=1000, value=6000, step=100, help="Increase max_tokens if no response is found", on_change=update_o1_model_tokens)  

    if choice == "Search Articles":
        st.subheader("Search Scientific Articles")
        query = st.text_input("Enter your search query", key='search_query')
        col1_search, col2_search = st.columns([1,1])
        with col1_search:
            max_results = st.number_input("Max results", key='search_max_results', min_value=1, max_value=20, value=5, step=1)
        with col2_search:
            option_sort_by = st.selectbox("Sort by", ['Relevance', 'Last Updated Date', 'Submitted Date'], key='search_sort_by')
        if st.button("Search"):
            results = search_arxiv(query, max_results, option_sort_by)
            st.session_state.results_arxiv = results
        if st.session_state.results_arxiv:
            st.markdown("### Results")
            results = st.session_state.results_arxiv
            for i, result in enumerate(results):
                with st.expander(f"**{str(i+1)}. {result['title']}**"):
                    st.write(f"**Authors**: {result['authors']}")
                    # st.write(f"**Categories**: {result['categories']}")
                    st.write(f"**Published**: {result['published']}")
                    st.write(f"**Updated**: {result['updated']}")
                    st.write(f"**Summary**: {result['summary']}")
                    st.write(f"[PDF Link]({result['url']})")
                # st.write("---")


    elif choice == "PDF Analysis":
        st.subheader("PDF Analysis")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            if st.session_state.pdf_analysis == "":
                analysis = analyze_pdf(uploaded_file)
            else:
                analysis = st.session_state.pdf_analysis
            st.write("PDF Analysis:")
            st.write(analysis)

    elif choice == "Hypothetical Experiments":
        st.subheader("Create Hypothetical Experiments")
        topic = st.text_input("Enter a topic for the hypothetical experiment", key='exp_topic')
        if st.button("Generate Experiment"):
            if st.session_state.experiment == "":
                experiment = create_hypothetical_experiment(topic, st.session_state.exp_max_tokens)
            else:
                experiment = st.session_state.experiment
            st.write("Hypothetical Experiment:")
            st.write(experiment)


    elif choice == "Reasoning Chat":
        st.subheader("Reasoning Chat")
        
        for message in st.session_state.chat_messages:
            if message["role"] != 'system':
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        print(st.session_state.chat_messages)

        if prompt := st.chat_input("What is up?"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Prepare the full history to send to the model
            full_history = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.chat_messages]

            # Analyze the chat history and select the appropriate model
            model_choice = analyze_chat_history(full_history)
            print(full_history)
            print(model_choice)

            # Send the full history to the appropriate model
            if model_choice == "o1-mini":
                response = gpt_o1_mini.chat(full_history)
                # print(response)
            else:
                response = gpt_4o_mini.chat(full_history)
                # print(response)

            response_text = str(response).replace('assistant: ', '', 1)
            # print(response['content'])

            if response_text == '':
                response_text = "Error: There are not enough tokens to complete it"

            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(f"{str(response_text).replace('assistant: ', '', 1)}\n\n**Model Used:** {model_choice}")

if __name__ == "__main__":
    main()