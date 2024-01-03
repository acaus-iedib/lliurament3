import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo serializado
with open('models/sc.pck', 'rb') as a:
    sc = pickle.load(a)

with open('models/m_logistic.pck', 'rb') as model_file:
    logistic = pickle.load(model_file)

with open('models/m_svm.pck', 'rb') as f:
        svm = pickle.load(f)     
        
with open('models/m_tree.pck', 'rb') as f:
        tree = pickle.load(f)           

with open('models/m_knn.pck', 'rb') as f:
        knn = pickle.load(f) 

label_mapping = {0: 'Iris Setosa', 1: 'Iris Versicolour', 2: 'Iris Virginica'}    

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
       
        data = request.get_json(force=True)
        # Agafem les dades dels atributs del petal
        input_data = [[data['apetal'], data['lpetal']]]
        #normalizacio
        input_data_normalized = sc.transform(input_data)
        #Predició
        prediction = logistic.predict(input_data_normalized)
        nom_flor = label_mapping[prediction[0]]
        return jsonify(nom_flor)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
       
        data = request.get_json(force=True)
        # Agafem les dades dels atributs del petal
        input_data = [[data['apetal'], data['lpetal']]]
        #normalizacio
        input_data_normalized = sc.transform(input_data)
        #Predició
        prediction = svm.predict(input_data_normalized)
        nom_flor = label_mapping[prediction[0]]
        return jsonify(nom_flor)

@app.route('/predict_tree', methods=['POST'])
def predict_tree():
       
        data = request.get_json(force=True)
        # Agafem les dades dels atributs del petal
        input_data = [[data['apetal'], data['lpetal']]]
        #normalizacio
        input_data_normalized = sc.transform(input_data)
        #Predició
        prediction = tree.predict(input_data_normalized)
        nom_flor = label_mapping[prediction[0]]
        return jsonify(nom_flor)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
        data = request.get_json(force=True)
 
        # Agafem les dades dels atributs del petal
        input_data = [[data['apetal'], data['lpetal']]]
        #normalizacio
        input_data_normalized = sc.transform(input_data)
        #Predició
        prediction = knn.predict(input_data_normalized)
        nom_flor = label_mapping[prediction[0]]

        return jsonify(nom_flor)

if __name__ == '__main__':
    app.run(debug=True, port=8000)  
