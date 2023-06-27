python -m venv venv
cd .\dochat\
.\venv\Scripts\activate
pip install -r .\requirements.txt
streamlit run app.py

python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"


https://github.com/alejandro-ao/ask-multiple-pdfs
https://www.youtube.com/watch?v=dXxQ0LR-3Hg&list=LL&index=11