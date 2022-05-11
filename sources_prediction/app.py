from flask import Flask, request, jsonify, g
from flasgger import Swagger
import os
import helpers
import sqlite3
import json

app = Flask(__name__)
swagger = Swagger(app)

DATABASE = 'database.db'


@app.route('/api/v1/models/create', methods=['POST'])
def create_model():  # put application's code here
    """Creating a new model from csv.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: dataset
        required: true
        type: file
        description: Upload your file.
    responses:
      200:
        description: Model created
    """
    uploaded_file = request.files['dataset']
    if uploaded_file.filename != '':
        filepath = os.path.join('..', 'models', 'datasets',
                                uploaded_file.filename)
        uploaded_file.save(filepath)
        dataset = helpers.load_dataset(uploaded_file.filename)
        df = helpers.prepare(dataset)
        m = helpers.fit(df)
        helpers.save_model(m, uploaded_file.filename)
    return 'Model created for ' + uploaded_file.filename


@app.route('/api/v1/models/<model>/update', methods=['PUT'])
def update_model(model):  # put application's code here
    """Update a model from csv.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: model
        in: path
        description: model name
        required: true
        type: string
      - in: formData
        name: dataset
        required: true
        type: file
        description: Upload your file.
    responses:
      200:
        description: Model created
    """
    uploaded_file = request.files['dataset']
    if uploaded_file.filename != '':
        filepath = os.path.join('..', 'models', 'datasets',
                                'update', uploaded_file.filename)
        uploaded_file.save(filepath)
        dataset = helpers.load_update_dataset(uploaded_file.filename)
        helpers.update_fitted_model(dataset, model)
    return 'Model updated for ' + uploaded_file.filename


@app.route('/api/v1/models/<model>/predict', methods=['GET'])
def predict(model):  # put application's code here
    """Predict the next interval for a model.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: model
        in: path
        description: model name
        required: true
        type: string
    responses:
      200:
        description: Prediction
    """
    m = helpers.load_model(model + '.csv')
    result = helpers.forecast(m, model, False)
    try:
        with sqlite3.connect("sources.db") as con:
            cur = con.cursor()
            cur.execute("INSERT OR REPLACE into Sources ("
                        "name, lower_bound, predicted, upper_bound) "
                        "values (?,?,?,?)", (model,
                                             result['lower_bound'],
                                             result['predicted'],
                                             result['upper_bound']))
            con.commit()
    except Exception as exc:
        con.rollback()
    finally:
        con.close()
    return jsonify(result)


@app.route('/api/v1/models/missing_data', methods=['GET'])
def delayed():  # put application's code here
    """Get all models which are delayed.
    ---
    responses:
      200:
        description: Delayed models
    """
    con = sqlite3.connect("sources.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from Sources where predicted < date('now')")
    rows = cur.fetchall()
    con.close()
    data = []
    for row in rows:
        data.append({'name': row[0],
                     'lower_bound': row[1],
                     'predicted': row[2],
                     'upper_bound': row[3]})
    return json.dumps(data)


def init_db():
    con = sqlite3.connect("sources.db")
    print("Database opened successfully")
    con.execute("create table if not exists Sources ("
                "name TEXT PRIMARY KEY, "
                "lower_bound TIMESTAMP NOT NULL, "
                "predicted TIMESTAMP NOT NULL, "
                "upper_bound TIMESTAMP NOT NULL)")
    print("Table created successfully")
    con.close()


if __name__ == '__main__':
    init_db()
    app.run(port=18888)
