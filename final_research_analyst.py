import os
import time
import hashlib
import sqlite3
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import traceback
from langchain.globals import set_verbose
from dotenv import load_dotenv
# Load environment variables
load_dotenv()


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('finobot.log', maxBytes=10000, backupCount=3)
logger.addHandler(handler)



# Database setup
conn = sqlite3.connect('finobot.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS processed_urls
             (url TEXT PRIMARY KEY, session_id TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS chat_history
             (id INTEGER PRIMARY KEY, role TEXT, message TEXT, session_id TEXT)''')
conn.commit()



# Streamlit interface
st.set_page_config(page_title="News Research Tool", page_icon="📰", layout="wide")
st.markdown("""
    <style>
    .main-title {
        font-size: 60px;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: #F0F8FF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom:50px;
    }
    </style>
    <h1 class="main-title">FinoBot 📈: Financial News & Analysis</h1>
    """, unsafe_allow_html=True)




# Sidebar Settings
st.sidebar.markdown('<hr style="border:1px solid #ff4b4b; margin-top:2px; margin-bottom:2px;">', unsafe_allow_html=True)
st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 40px;
        font-weight: bold;
        color: #1E90FF;
        margin-bottom: 20px;
    }
    .sidebar-section {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        margin-top:5px;
    }
    </style>
    <div class="sidebar-title">⚙️ Settings</div>
    """, unsafe_allow_html=True)
st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;margin-top:2px;">', unsafe_allow_html=True)


# Session management
def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    return st.session_state.session_id

session_id = get_session_id()


# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = datetime.min



# Finobot Usage Guide
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f8ff;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown(f""" <h1 style="font-size:30px; color:#FFD700;">Quick Guide 🚀 </h1>""", unsafe_allow_html=True)
st.sidebar.markdown("""
1. 📥 **Input:** Enter up to 3 article URLs.
2. ⚙️ **Configure:** Set Chunk Size and Temperature.
3. ▶️ **Process:** Click "Process URLs".
4. ❓ **Query:** Ask your question in main area.
5. 📊 **Review:** Examine answer.
6. 🔄 **Manage:** Clear data or export history as needed.

**Note: Process URLs before querying.**
""")


# Sample links
st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
st.sidebar.markdown("""
        <h1 style="font-size:30px; color:#FFD700;">Sample Links 🌐 </h1>
        <p style="font-size:20px; color:#4169e1;">Sample links for testing purpose 📄 </p>
        <ul>
            <li><a href="https://www.moneycontrol.com/news/podcast/can-markets-sustain-its-rally-after-mounting-to-new-highs-market-minutes-12781742.html" style="color: red;">🔗 Sample Link 1</a></li>
            <li><a href="https://www.moneycontrol.com/news/business/markets/govt-may-raise-stcg-above-20-in-future-official-12782916.html" style="color: red;">🔗 Sample Link 2</a></li>
            <li><a href="https://www.moneycontrol.com/news/business/itr-filing-last-date-income-tax-deadline-extension-july-2024-live-news-updates-liveblog-12782752.html" style="color: red;">🔗 Sample Link 3</a></li>
        </ul>
 
    """, unsafe_allow_html=True)



# URL section
st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
st.sidebar.markdown(f"""<h1 style="font-size:30px; color:#FFD700;">URLs Section 🔗</h1>""", unsafe_allow_html=True)
# st.sidebar.markdown('<h3 style="color: #1E90FF;">🔗 URLs Section</h3>', unsafe_allow_html=True)

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except:
        return False


# URL inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        if validate_url(url):
            urls.append(url)
        else:
             st.sidebar.warning(f"⚠️ URL {i+1} is not valid. Please enter a valid http or https URL.")
st.sidebar.markdown('</div>', unsafe_allow_html=True)




# Customization options
st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
st.sidebar.markdown(f"""<h1 style="font-size:30px; color:#FFD700;">Customization 🛠️</h1>""", unsafe_allow_html=True)

chunk_size = st.sidebar.slider("📏 Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
temperature = st.sidebar.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# process_url_clicked = st.sidebar.button("Process URLs")
process_url_clicked = st.sidebar.button("▶️ Process URLs", key="process_urls")





# Main content area
main_placeholder = st.empty()


# Initialize LLM
try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=500)
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    logger.error(f"LLM initialization error: {str(e)}\n{traceback.format_exc()}")
    st.stop()

def process_urls(urls):
    new_urls = [url for url in urls if url not in [row[0] for row in c.execute("SELECT url FROM processed_urls WHERE session_id=?", (session_id,))]]
    
    if not new_urls:
        st.warning("No new valid URLs to process.")
        return

    try:
        with st.spinner('Processing URLs...'):
            loader = UnstructuredURLLoader(urls=new_urls)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=chunk_size
            )
            docs = text_splitter.split_documents(data)
            
            try:
                embeddings = OpenAIEmbeddings()
            except Exception as e:
                st.warning("Failed to initialize OpenAI embeddings. Falling back to default embeddings.")
                logger.warning(f"OpenAI embeddings initialization failed: {str(e)}")
                embeddings = FAISS.default_embeddings()

            vectorstore = FAISS.from_documents(docs, embeddings)
            
            if st.session_state.vectorstore:
                st.session_state.vectorstore.merge_from(vectorstore)
            else:
                st.session_state.vectorstore = vectorstore
            
            for url in new_urls:
                c.execute("INSERT OR REPLACE INTO processed_urls (url, session_id) VALUES (?, ?)", (url, session_id))
            conn.commit()
        
        st.success(f"Processed {len(new_urls)} new URLs.")
    except Exception as e:
        st.error(f"An error occurred while processing URLs: {str(e)}")
        logger.error(f"URL processing error: {str(e)}\n{traceback.format_exc()}")

if process_url_clicked:
    process_urls(urls)


st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #1E90FF;
    }
    .stButton > button {
        background-color: #1E90FF;
        color: white;
        font-size: 1.1rem;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #4169E1;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"""<h1 style="font-size:40px; color:#FFD700; margin-bottom: 1px;">❓ Ask a question about the processed news articles:</h1>""", unsafe_allow_html=True)
st.markdown('<hr style="border:1px solid #ff4b4b;margin-top:2px;">', unsafe_allow_html=True)
query = st.text_input('-----------------------------------------------------')
ask_button = st.button('🔍 Ask Question')


if ask_button and query:
    if datetime.now() - st.session_state.last_query_time < timedelta(seconds=5):
        st.warning("Please wait a few seconds before submitting another query.")
    elif st.session_state.vectorstore is not None:
        try:
            qa_with_sources_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever()
            )
            
            with st.spinner('Searching for an answer...'):
                answer = qa_with_sources_chain.invoke({"question": query})
            
            st.header("💡 Answer")
            st.write(answer['answer'])
            
            # st.header("📚 Sources")
            # sources = answer['sources'].split('\n')
            # for source in sources:
            #     if source.strip():
            #         st.markdown(f"- {source}")


            
            # Add to chat history
            c.execute("INSERT INTO chat_history (role, message, session_id) VALUES (?, ?, ?)", ("You", query, session_id))
            c.execute("INSERT INTO chat_history (role, message, session_id) VALUES (?, ?, ?)", ("AI", answer['answer'], session_id))
            conn.commit()

            st.session_state.last_query_time = datetime.now()
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            logger.error(f"Question processing error: {str(e)}\n{traceback.format_exc()}")
    else:
        st.error("Please process URLs first before asking questions.")
elif ask_button:
    st.warning("Please enter a question.")


# Display Chat History
st.markdown("""
    <style>
    .chat-history {
       /* # background-color: #F0F8FF;*/
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .chat-history h2 {
        color: #FFD700;
        margin-bottom: 1px;
        font-size:40px;
    }
    </style>
    <div class="chat-history">
    <h2>Chat History 📜</h2>
    
    </div>
    <hr style="border:1px solid #ff4b4b;margin-top:1px;">
    """, unsafe_allow_html=True)

for row in c.execute("SELECT role, message FROM chat_history WHERE session_id=? ORDER BY id", (session_id,)):
    st.markdown(f"**{row[0]}:** {row[1]}")


# Clear data button
if st.sidebar.button("🗑️ Clear Processed Data", key="clear_data"):
# if st.sidebar.button("Clear Processed Data"):
    try:
        st.session_state.vectorstore = None
        c.execute("DELETE FROM processed_urls WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
        conn.commit()
        st.sidebar.success("All processed data has been cleared. You can now start fresh.")
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
        logger.error(f"Data clearing error: {str(e)}\n{traceback.format_exc()}")




# Display processed URLs
processed_urls = [row[0] for row in c.execute("SELECT url FROM processed_urls WHERE session_id=?", (session_id,))]
if processed_urls:
    st.sidebar.markdown(f"""<h1 style="font-size:20px; color:#ff4b4b;">Processed URLs</h1>""", unsafe_allow_html=True)
    # st.sidebar.header("Processed URLs")
    for url in processed_urls:
        st.sidebar.markdown(f"- {url}")



# Export chat history
# Styling for the chat history section
st.markdown("""
    <style>
    .chat-export-section {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        color:#1e90ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-export-title {
        color: #ff4b4b;
        font-size: 24px;
        margin-bottom: 15px;
    }
    .stDownloadButton>button {
        background-color: #1e90ff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stDownloadButton>button:hover {
        background-color: #4169e1;
    }
    </style>
""", unsafe_allow_html=True)

# Chat history export section
st.markdown('<h3 class="chat-export-title">📥 Export Chat History</h3>', unsafe_allow_html=True)

if st.button("Prepare Export 🔄"):
    try:
        with st.spinner("Preparing chat history..."):
            chat_history = c.execute("SELECT role, message FROM chat_history WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
            chat_history_text = "\n\n".join([f"{'🧑‍💼' if role == 'You' else '🤖'} {role}:\n{message}" for role, message in chat_history])
        
        st.success("Chat history prepared successfully! ✅")
        st.download_button(
            label="📥 Download Chat History",
            data=chat_history_text,
            file_name="finobot_chat_history.txt",
            mime="text/plain",
            key="download_chat"
        )
    except Exception as e:
        st.error(f"🚫 Error exporting chat history: {str(e)}")
        logger.error(f"Chat history export error: {str(e)}\n{traceback.format_exc()}")

st.markdown('', unsafe_allow_html=True)



# Feedback Section
st.sidebar.markdown('<hr style="border:1px solid #ff4b4b;">', unsafe_allow_html=True)
st.sidebar.markdown(f"""
    <h1 style="font-size:30px; color:#FFD700;">Feedback 📝</h1>
    <p>We value your feedback! 😊</p>
""", unsafe_allow_html=True)

feedback = st.sidebar.text_area("Please leave your feedback here:")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback! 👍")



# Close database connection
conn.close()


# Footer
st.markdown("---")
st.markdown('<p style="text-align: center;">Created with ❤️ using Streamlit and LangChain</p>', unsafe_allow_html=True)