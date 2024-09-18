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
reddit = praw.Reddit(client_id='FqSMlY_FfDLybf8QT8jKOA',
                     client_secret='HDYjkUEDIbdOR_G6AX48sZ0FYZL_3g',
                     user_agent='owl-99')

# Set the Groq API key
os.environ['GROQ_API_KEY'] = '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'

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
        if 'url' in request.form:
            url = request.form['url']
            post = reddit.submission(url=url)
            comments = post.comments.list()

            pdf_path = create_pdf(post, comments)
            flash(f'PDF saved successfully at: {pdf_path}')  # Flash message to indicate success

            # Store the PDF path in session
            session['pdf_path'] = pdf_path

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

        elif 'user_prompt' in request.form:
            user_prompt = request.form['user_prompt']
            rag_tool = PDFSearchTool(
                pdf=session['pdf_path'],  # Pass the PDF path directly
                config=session['rag_tool_config']  # Pass the configuration
            )
            outputt = rag_tool.run(user_prompt)  # Ensure this is the correct variable

            # Check if outputt is valid
            if not outputt:
                flash('Error: No output generated from the PDF search tool.')
                return render_template('index.html')

            # Define CrewAI agent with required fields
            agent = Agent(
                role="Assistant",  # Specify the role
                goal="Just display the input data you have received.",  # Specify the goal
                backstory="Just display the input data you have received.",  # Specify the backstory
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )
            
            # Define the task with required fields
            task = Task(
                description="Just display the input data you have received.",  # Add description
                expected_output="Just display the input data you have received.",
                agent=agent,  # Add expected output,
            )
            
            # Create the Crew instance with the correct input
            rag_crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
            )
            print("outputt:::", outputt)

            inputs1 = {"question": outputt}
            print(inputs1)

            # Execute the Crew instance
            pain_points = rag_crew.kickoff(inputs=inputs1)

            print(f"Output from PDFSearchTool: {outputt}")
            print(f"Pain Points from CrewAI: {pain_points}")

            return render_template('index.html', output=outputt, pain_points=pain_points)  # Pass pain points to the template

    return render_template('index.html')

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
