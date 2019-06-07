#Import Flask
from flask import Flask, request
from keras.preprocessing import image
from ann_loader import cargarModelo
import numpy as np

#Initialize the application service
app = Flask(__name__)
global loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2, graph
loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - Curso de RNA en los negocios!'

@app.route('/abandono/', methods=['GET','POST'])
def churn():
	return 'Modelo de Abandono de Clientes!'

@app.route('/abandono/cliente/', methods=['GET','POST'])
def default():

	print (request.args)

	# Obtenendo parametros
	scoreCrediticio = request.args.get("scoreCrediticio")
	pais = request.args.get("pais")
	genero = request.args.get("genero")
	edad = request.args.get("edad")
	tenencia = request.args.get("tenencia")
	balance = request.args.get("balance")
	numDeProductos = request.args.get("numDeProductos")
	tieneTarjetaCredito = request.args.get("tieneTarjetaCredito")
	esMiembroActivo = request.args.get("esMiembroActivo")
	salarioEstimado = request.args.get("salarioEstimado")

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
		resultado = "Predicción: "
		score = loaded_model.predict(cliente)
		print("\nFinal score: ", score)
		abandona = (score > 0.5)
		if abandona:
			resultado += "Abandona"
		else:
		    resultado += "No abandona"
		return resultado + ', score: ' + str(score)

	# http://localhost:5000/abandono/cliente/?scoreCrediticio=3&pais=France&genero=Male&edad=36&tenencia=2&balance=1200.34&numDeProductos=3&tieneTarjetaCredito=1&esMiembroActivo=0&salarioEstimado=120000
	# http://localhost:5000/abandono/cliente/?scoreCrediticio=1&pais=Spain&genero=Female&edad=50&tenencia=2&balance=200.34&numDeProductos=1&tieneTarjetaCredito=0&esMiembroActivo=0&salarioEstimado=85000

# Run de application
app.run(host='0.0.0.0',port=5000)
