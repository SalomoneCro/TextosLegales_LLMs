from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import load_checkpoint_and_dispatch
import os
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("SweatyCrayfish/llama-3-8b-quantized")

# Create a BitsAndBytesConfig for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model with the BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    "SweatyCrayfish/llama-3-8b-quantized",
    quantization_config=quantization_config,
    device_map="auto"
)

# Step 3: Create the summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

jud_text = """
OF. COBROS PARTICULARES (JUZG.1a
Nom)-MARCOS JUAREZ
Protocolo de Autos
Nº Resolución: 98
Año: 2022 Tomo: 1 Folio: 105-107

EXPEDIENTE SAC: 7103025 - RUFFINO, ROBERTO RAUL C/ MATALIA, JAVIER ALEJANDRO - CUERPO DE EJECUCION
- CUERPO DE EJECUCIÓN
PROTOCOLO DE AUTOS. NÚMERO: 98 DEL 27/05/2022

AUTO NUMERO: 98.
MARCOS JUAREZ, 27/05/2022.

VISTOS: Estos autos caratulados “RUFFINO, ROBERTO RAUL C/ MATALIA,
JAVIER ALEJANDRO - CUERPO DE EJECUCION”,(Expte. Nº 7103025), de los que
resulta que:
I.- Con fecha 18/03/2022, el Dr. Matías José PRATTI comparece y solicita se libre orden de
pago en concepto de capital a favor de su poderdante, Sr. ROBERTO RAUL RUFFINO,
D.N.I. nº 16.515.479, por la suma de DÓLARES ESTADOUNIDENSES VEINTISIETE MIL
TREINTA Y SIETE CON C/08/100 (U$27.037,08.-) en concepto de capital e intereses.
Manifiesta que en la Sentencia N° 11 de fecha 10/02/2017, no se fijaron los parámetros de
conversión, puesto que se resolvió que el demandado debía pagar en dólares estadounidenses.
Sin embargo, el producido de la subasta corresponde a pesos y no dólares, por lo que
necesariamente deberá realizarse la conversión de pesos a dólares estadounidenses a los fines
del cobro.
En dicho sentido, expresa que tomar el promedio del dólar "oficial" que fija el Banco Nación
para la compra y para la venta, beneficiaria al incumplidor, y afectaría el derecho de

Expediente SAC 7103025 - Pág. 1 / 5 - Nº Res. 98


propiedad del actor, vulnerando el principio rector de buena fe consagrado por los arts. 961 y
9 del Código Civil y Comercial de la Nación, arribando a una solución que dista de ser justa.Cita Jurisprudencia en apoyo.
Entonces, propone se tome el dólar MEP (Dólar Mercado Electrónico de Pagos), tal como lo
fijó -entre otros- la Cámara de Apelaciones en lo Civil y Comercial del Mar del Plata o en su
defecto, al pago de la suma convertida en pesos al cambio del día del efectivo pago, con más
el 30% correspondiente al art. 35 Ley 27541 (impuesto al ahorro - dólar solidario), con más
intereses acordes a los que corresponden a préstamos bancarios en dólares todo hasta el
efectivo pago. Cita Jurisprudencia.
Manifiesta, que lo que se pretende es una solución justa, que no beneficie al deudor, pues
hace desde el año 2017 que el actor persigue el cobro de la deuda de autos, y que no se
desvirtúe la propia sentencia que ordenó que se entreguen dólares, pues es de público y
notorio conocimiento que al valor del dólar “oficial” resulta imposible adquirir dólares
billetes, de hecho la cotización al día de la fecha del dólar “oficial” es: $108,50 comprador y
$114,50 vendedor; el dólar MEP es: $195,59 comprador y $195,90 vendedor; y el dólar
solidario $188,93 (fuente: https://www.cronista.com/MercadosOnline/dolar.html), lo cual deja
en evidencia la disparidad cambiaria que existe.
II.- De lo expuesto por la parte actora, con fecha 04/04/2022 se ordena dar noticia a la
contraria. Vencido el plazo para ello, sin ser evacuada, se dicta el decreto de autos con fecha
27/04/2022, el que firme, deja la presente incidencia en estado de ser resuelta.
Y CONSIDERANDO:
I. Que la parte actora al solicitar se libre orden de pago en concepto de capital e intereses pide
convertir la suma de pesos depositada, como resultado de la subasta producida en autos, al
tipo de cambio que resulte más justo y que no beneficie al deudor. A tales fines, propone para
ello tomar el dólar MEP o la suma convertida en pesos al cambio del día del efectivo pago,
con más el 30% correspondiente al art. 35 Ley 27541 (impuesto al ahorro - dólar solidario),
Expediente SAC 7103025 - Pág. 2 / 5 - Nº Res. 98


con más intereses acordes a los que corresponden a préstamos bancarios en dólares todo hasta
el efectivo pago. La contraria no contesta la noticia corrida y en estos términos queda la
cuestión a resolver.
II. En primer lugar, cabe destacar que la situación actual, donde además de la restricción para
la adquisición de la moneda extranjera, se derivan distintos desdoblamientos y cotizaciones de
cambio, obliga a un mayor rigor en la determinación de la moneda de pago, para evitar
inequidades. Estas restricciones cambiarias han implicado la imposibilidad de la adquisición
de moneda para el pago total de la deuda (decreto 609/2019, Resolución 6770 BCRA y
conc.).
En consecuencia, considero que debe establecerse una cotización más próxima a la realidad
en el contexto económico de nuestro país.
Ahora bien, puestos a determinar la cotización que resulta adecuada, si se tiene en
consideración el contexto financiero actual en el que existen restricciones que limitan la
adquisición de la señalada moneda extranjera (Comunicación BCRA A6815 y cc), gravada
además con el impuesto PAIS e “Impuesto para una Argentina Inclusiva y Solidaria” (ley
27.541), es evidente que la conversión de los dólares a la cotización oficial no arroja una
suma “equivalente” en pesos que satisfaga el interés del acreedor o resulte justa, ya que con
esa cantidad de pesos este no podría adquirir en el mercado de cambios la suma de dólares
adeudada y que por tal motivo se pactó en billete.
Aclárese que la alícuota del 30% adicional derivado de la aplicación de este impuesto, no es
un componente del valor de la divisa sino, precisamente, un tributo (conf. CALDERON,
Maximiliano, LA LEY, Cita Online: AR/LEGI/9Z02), por lo que resulta de más complicado
adicionar al valor de cotización de la moneda extranjera el impuesto PAÍS y, más aún, el
anticipo a cuenta del impuesto a las ganancias y bienes personales reglamentado por la
resolución general de la AFIP n°4815/2020.
En ese orden de ideas, y en concordancia con lo dispuesto por el Juzgado de Primera Instancia
Expediente SAC 7103025 - Pág. 3 / 5 - Nº Res. 98


y Tercera Nominación- Secretaría Quinta en lo Civil, Comercial y Familia de la Ciudad de
Rio Cuarto, Sentencia N° 7 del 24/02/2022 en los autos caratulados L., A. L. – PEQUEÑO
CONCURSO PREVENTIVO, dentro del abanico que otorga el mercado cambiario legal y
regulado, la cotización del denominado dólar “MEP” (mercado electrónico de pagos) resulta
adecuada.
Para concluir de ese modo se tiene en cuenta que su precio deriva de la compra y venta de
títulos públicos (con las regulaciones específicas), de conformidad con los valores propios del
mercado y sin afectar las reservas públicas con cotización que puede ser conocida por el
público por medio de las diferentes vías de información periodística, lo cual otorga publicidad
y transparencia a tal valor de conversión (conf., CNCiv., Sala M, “Bazo, Susana C. c/ Cano
Vázquez, Horacio E. s/ ejecución”, 29/04/21; Id., Id., “Tobio Romero, José c/ Tursi, María
Rita s/ ejecución de honorarios mediación”, 18/02/21; CNCom, Sala D, voto del juez Vasallo,
“Ortola Martínez, Gustavo Marcelo c. Sarlenga, Marcela Claudia s/Ordinario”, del
15/10/2020, La Ley Online, AR/JUR/47237/2020; CNC, Sala J, 20/05/2021, Expte. n°63721/
2015,“Nakkab, Sion Gabriel c/Roccasalvo, Ricardo Daniel y otro s/ División de
condominio”).
Destáquese que el dólar MEP es aquel tipo de cambio resultante de una operación sencilla
que consiste en la compra de bonos en pesos y su posterior venta en dólares, lo que permite
comprar dólares sin límite por mes dado que la operación no está alcanzada por el impuesto
del 30% (impuesto PAIS), por lo que entiendo se garantiza el principio de buena fe, la
interdicción del abuso del derecho y más aún, el pago íntegro de la deuda de autos, liberando
al deudor de su obligación “dando el equivalente en moneda de curso legal”.
En suma, las partes deberán adecuar las cuentas a la cotización del promedio tipo
comprador y tipo vendedor del dólar MEP a la fecha del efectivo libramiento de la orden
de pago requerida en concepto de capital e intereses. A modo indicativo, a la fecha se ubica
en $ 210,35 para la compra y $ 213,09 para la venta, por lo que el promedio equivale a $
Expediente SAC 7103025 - Pág. 4 / 5 - Nº Res. 98


211.72 ( https://www.cronista.com/finanzas-mercados/dolar-blue-hoy-jueves-26-de-mayo-acuanto-cotiza/).
III.- Costas: A mérito de la forma en que se decide la presente incidencia, donde no existió
oposición de la contraria, y en virtud de la divergencia doctrinaria y jurisprudencial sobre el
tema sometido a decisión, considero justo y equitativo imponer las costas por el orden
causado (arg. art. 130, última parte, C. de P. C. -al cual remite el art. 133, ib.).
Por las razones expuestas y normas legales citadas.
RESUELVO:
I) Establecer que el monto por el que prosperó la ejecución en concepto de capital e intereses
conforme última liquidación aprobada en autos con fecha 04/04/2022, deberá convertirse a
moneda de curso legal, según la cotización del promedio tipo comprador y tipo vendedor
del dólar MEP a la fecha del efectivo libramiento de la orden de pago requerida por dichos
rubros.
II) Sin costas.
Protocolícese y dese copia.-

Texto Firmado digitalmente por:

TONELLI Jose Maria
JUEZ/A DE 1RA. INSTANCIA
Fecha: 2022.05.27

Expediente SAC 7103025 - Pág. 5 / 5 - Nº Res. 98
"""

def split_text(text, chunk_size):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    # Split tokens into chunks of chunk_size
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    # Convert token chunks back into text
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return text_chunks

start_time = time.time()

# Step 6: Summarize each chunk individually
chunks = split_text(jud_text, chunk_size=500)  # Split the text into chunks of 400 tokens

#Forma chatGPT
# summaries = [summarizer(chunk, max_new_tokens=150, min_new_tokens=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text'] for chunk in chunks]
# final_summary = " ".join(summaries)

'''
max_new_tokens: Define la longitud máxima del resumen.
mix_new_tokens: Establece la longitud mínima del resumen.
length_penalty=2.0: Controla el sesgo hacia resúmenes más largos o cortos.
num_beams=4: Utiliza búsqueda por haz (beam search) para mejorar la calidad del resumen.
early_stopping=True: Detiene la generación si el modelo cree que ha llegado a un buen resultado antes de alcanzar la longitud máxima.
'''


# Forma Pedro
for i in range(len(chunks) - 1):
    summ = summarizer(chunks[i], max_new_tokens=400, min_new_tokens=250, num_beams=4)[0]['summary_text']
    iteration_summ = " ".join([summ, chunks[i+1]])

summary = summarizer(iteration_summ, max_new_tokens=1500, min_new_tokens=1000, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']
print(f"Tokens primer resumen: {len(tokenizer(summary)['input_ids'])}")
summary = summarizer(iteration_summ, max_new_tokens=800, min_new_tokens=500, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']
print(f"Tokens segundo resumen: {len(tokenizer(summary)['input_ids'])}")


finish_time = time.time()

print(summary)

print(f"Se tardo {(finish_time - start_time) / 60} minutos en generar el resumen")