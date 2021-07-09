from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sys
import os

app = Flask(__name__, template_folder='./templates/')


@app.before_first_request
def at_startup():

    global answer_df, question_map, top_k_map

    answer_df = pd.read_csv("./all_question_comprehension.csv", index_col=None)
    question_map = {'1': 'Is the virus transmitted by aerosol, droplets, food, close contact, fecal matter, or water?',
                    '2': 'How long is the incubation period for the virus?',
                    '3': 'Can the virus be transmitted asymptomatically or during the incubation period?',
                    '4': 'How does weather, heat, and humidity affect the transmission of 2019-nCoV?',
                    '5': 'How long can the 2019-nCoV virus remain viable on common surfaces?',
                    '6': 'What risk factors contribute to the severity of 2019-nCoV?',
                    '7': 'How does hypertension affect patients?',
                    '8': 'How does heart disease affect patients?',
                    '9': 'How does copd affect patients?',
                    '10': 'How does smoking affect patients?',
                    '11': 'How does pregnancy affect patients?',
                    '12': 'What is the fatality rate of 2019-nCoV?',
                    '13': 'What public health policies prevent or control the spread of 2019-nCoV?',
                    '14': 'Can animals transmit 2019-nCoV?',
                    '15': 'What animal did 2019-nCoV come from?',
                    '16': 'What real-time genomic tracking tools exist?',
                    '17': 'What geographic variations are there in the genome of 2019-nCoV?',
                    '18': 'What efforts are being done in asia to prevent further outbreaks?',
                    '19': 'What drugs or therapies are being investigated?',
                    '20': 'Are anti-inflammatory drugs recommended?',
                    '21': 'Which non-pharmaceutical interventions limit transmission?',
                    '22': 'What are the most important barriers to compliance?',
                    '23': 'How does extracorporeal membrane oxygenation affect 2019-nCoV patients?',
                    '24': 'What telemedicine and cybercare methods are most effective?',
                    '25': 'How is artificial intelligence being used in real time health delivery?',
                    '26': 'What adjunctive or supportive methods can help patients?',
                    '27': 'What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV?',
                    '28': 'What is the immune system response to 2019-nCoV?',
                    '29': 'Can personal protective equipment prevent the transmission of 2019-nCoV?',
                    '30': 'Can 2019-nCoV infect patients a second time?'}

    top_k_map = {'0': 5, '1': 10, '2': 20, '3': 30, '4': 50}


@app.route('/')
def home():
    return render_template("index.html")


def create_answer(text, start, end):
    output = [text[0:start],
              text[start:end],
              text[end:len(text)]]
    return output


@app.route('/top_k_results', methods=['GET', 'POST'])
def top_k_results():
    question_select = "0"
    weight = "0.2"
    top_k = "0"
    if request.method == "POST":
        question_select = request.form.get('question_select', '')
        weight = request.form.get('weight', '')
        top_k = request.form.get('top_k', '')

    query = question_map[question_select]
    # Filtering answer dataframe for the query
    _df = answer_df[answer_df['query'].isin([query])]
    _df = _df.drop_duplicates(subset=['passage_id']).reset_index(drop=True)

    _df["final_score"] = np.float(
        weight)*_df["score"] + (1-np.float(weight))*_df["pass_rank_score"]

    _df = _df.sort_values(
        'final_score', ascending=False).reset_index(drop=True)

    # results-dictionary
    results = [{'passage': create_answer(row['passage'], row['start'], row['end']),
                'title':row['title'],
                'task':row['task']} for i, row in _df.head(top_k_map[top_k]).iterrows()]

    return render_template("index.html", question_select=question_select,
                           weight=weight, top_k=top_k, results=results)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run('0.0.0.0', port)
