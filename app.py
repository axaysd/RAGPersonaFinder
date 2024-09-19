from flask import Flask, render_template, request, redirect, url_for, flash, session
import praw
from fpdf import FPDF
import os
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from crewai_tools import tool
from crewai import Crew
from crewai import Task
from crewai import Agent

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Configure Reddit API
reddit = praw.Reddit(client_id='',
                     client_secret='',
                     user_agent='')

# Set the Groq API key
os.environ['GROQ_API_KEY'] = ''

# Directory to save PDFs
PDF_DIR = 'pdfs'
os.makedirs(PDF_DIR, exist_ok=True)

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],  # Use the Groq API key
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Capture the creative_mode value
        creative_mode = request.form.get('creative_mode', 'off')  # Default to 'off' if not set

        if 'url' in request.form:
            # Existing logic for handling URL submission
            url = request.form['url']
            post = reddit.submission(url=url)
            comments = post.comments.list()

            pdf_path = create_pdf(post, comments)
            flash(f'PDF saved successfully at: {pdf_path}')  # Flash message to indicate success

            # Store the PDF path and creative_mode in session
            session['pdf_path'] = pdf_path
            session['creative_mode'] = creative_mode  # Store creative_mode in session

            # Store rag_tool configuration in session
            session['rag_tool_config'] = {
                'llm': {
                    'provider': 'groq',
                    'config': {
                        'model': 'llama3-8b-8192',
                    },
                },
                'embedder': {
                    'provider': 'huggingface',
                    'config': {
                        'model': 'BAAI/bge-small-en-v1.5',
                    },
                },
            }

            return render_template('index.html', rag_tool=True)  # Indicate rag_tool is ready

        # This block will execute regardless of the creative_mode value
        if 'creative_mode' in request.form or 'creative_mode' not in request.form:
            # Use the stored PDF path and creative_mode
            rag_tool = PDFSearchTool(
                pdf=session['pdf_path'],  # Pass the PDF path directly
                config=session['rag_tool_config']  # Pass the configuration
            )
            outputt = rag_tool.run(query=session['pdf_path'])  # Pass the PDF path or relevant query

            # Define CrewAI agent and task using creative_mode
            agent = Agent(
                role="Find out the pain points users mention in the product reviews by analyzing the user reviews from {outputt}.",
                goal="Analyze user reviews for the product and identify pain-points.",
                backstory="You have to look into {outputt} to identify the painpoints based on the reviews users have left about the product.",
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )
            
            task = Task(
                description="Find out the pain points users by analyzing the user reviews from {outputt}. Only if the value of {creative_mode} is on, generate some important pain points that you didn't find in {outputt} for the product. For the pain points that you generated yourself if the value of {creative_mode} was on, add a prefix 'X_'.",
                expected_output="A list of painpoints with the username(s) of the user(s) associated with the painpoint.",
                agent=agent,
                creative_mode=creative_mode  # Pass the creative_mode value
            )

            # Create the Crew instance and execute
            rag_crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
            )
            inputs1 = {"question": outputt, "outputt": outputt, "creative_mode": creative_mode}  # Add creative_mode to inputs1
            pain_points = rag_crew.kickoff(inputs=inputs1)
            print("creative_mode:", creative_mode)

            return render_template('index.html', output=outputt, pain_points=pain_points, rag_tool=True)

    return render_template('index.html', rag_tool=False)  # Default render

def create_pdf(post, comments):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Function to safely encode text
    def safe_text(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Add post content
    pdf.cell(200, 10, txt=safe_text(f"Title: {post.title}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"Author: {post.author}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"Content: {post.selftext}"), ln=True)
    pdf.cell(200, 10, txt="Comments:", ln=True)

    # Add comments
    for comment in comments:
        pdf.cell(200, 10, txt=safe_text(f"{comment.author}: {comment.body}"), ln=True)

    pdf_file_path = os.path.join(PDF_DIR, 'reddit_post.pdf')
    pdf.output(pdf_file_path)
    return pdf_file_path

if __name__ == '__main__':
    app.run(debug=True)
