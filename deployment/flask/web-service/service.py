import os
#Import Flask
from flask import Flask, request
from flask_cors import CORS
from keras.preprocessing import image
from ann_loader import cargarModelo
import numpy as np

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 5000
port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2, graph
loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Modelo desplegado en la Nube!'

@app.route('/abandono/', methods=['GET','POST'])
def churn():
	return 'Modelo de Abandono de Clientes!'

@app.route('/abandono/cliente/', methods=['GET','POST'])
def default():
	# print (request.data)
	# print (request.args)
	# print (request.form)
	data = None
	if request.method == 'GET':
		print ("GET Method")
		data = request.args

	if request.method == 'POST':
		print ("POST Method")
		if (request.is_json):
			data = request.get_json()

	print("Data received:", data)

	# Obteniendo parametros
	scoreCrediticio = data.get("scoreCrediticio")
	pais = data.get("pais")
	genero = data.get("genero")
	edad = data.get("edad")
	tenencia = data.get("tenencia")
	balance = data.get("balance")
	numDeProductos = data.get("numDeProductos")
	tieneTarjetaCredito = data.get("tieneTarjetaCredito")
	esMiembroActivo = data.get("esMiembroActivo")
	salarioEstimado = data.get("salarioEstimado")

	print ("\nscoreCrediticio: ",scoreCrediticio,
			"\npais: ", pais,
			"\ngenero: ", genero,
			"\nedad: ", edad,
			"\ntenencia: ", tenencia,
			"\nbalance: ", balance,
			"\nnumDeProductos: ", numDeProductos,
			"\ntieneTarjetaCredito: ", tieneTarjetaCredito,
			"\nesMiembroActivo: ", esMiembroActivo,
			"\nsalarioEstimado: ", salarioEstimado)

	# Transformado/Escalando la data
	[pais] = loaded_labelEncoderX1.transform([pais])
	[genero] = loaded_labelEncoderX2.transform([genero])

	cliente = np.array([scoreCrediticio,pais,genero,edad,tenencia,balance,numDeProductos,tieneTarjetaCredito,esMiembroActivo,salarioEstimado])
	print("\ncliente: ", cliente)
	cliente = loaded_scaler.transform([cliente])
	print("cliente Norm: ", cliente)

	with graph.as_default():
		resultado = ""
		score = loaded_model.predict(cliente)
		print("\nFinal score: ", score)
		abandona = (score > 0.5)
		if abandona:
			resultado += "Abandona"
		else:
		    resultado += "No abandona"
		return resultado + ', score: ' + str(score[0])

	# http://localhost:5000/abandono/cliente/?scoreCrediticio=3&pais=France&genero=Male&edad=36&tenencia=2&balance=1200.34&numDeProductos=3&tieneTarjetaCredito=1&esMiembroActivo=0&salarioEstimado=120000
	# http://localhost:5000/abandono/cliente/?scoreCrediticio=1&pais=Spain&genero=Female&edad=50&tenencia=2&balance=200.34&numDeProductos=1&tieneTarjetaCredito=0&esMiembroActivo=0&salarioEstimado=85000

# Run de application
app.run(host='0.0.0.0',port=port)
