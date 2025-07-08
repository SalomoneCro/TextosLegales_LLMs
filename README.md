# Resumen Automático de Fallos Judiciales con LLMs

## Introducción
En el Boletín Judicial de los Tribunales Provinciales de Córdoba, un equipo liderado por la Dra. Valeria Sola trabaja en la extracción de metadatos, redacción de síntesis y elaboración de sumarios de fallos judiciales. Estas tareas se realizan manualmente sobre textos denominados [Fallos](https://drive.google.com/file/d/1M7qSGdbUqznr94c2qDan3uIS5SkR5TiE/view?usp=sharing), generando como resultado documentos denominados [Sumarios](https://drive.google.com/file/d/1yk6CjVThjc6iFP7Dw3XlPv6srfCkQhDi/view?usp=sharing). El objetivo es ofrecer documentos estandarizados que resumen causas legales en los fueros Civil y Penal, siguiendo las [Normas de estilo para la redacción de sumarios de jurisprudencia](https://drive.google.com/file/d/18NEBdtVR5UuGBziGEfFb1CA8ePUUsOnd/view?usp=sharing).

Este proceso manual es altamente demandante y consume tiempo que los abogados podrían dedicar a tareas más interpretativas. Este proyecto propone una herramienta automática que genere textos preliminares, permitiendo a los abogados centrarse en la validación, profundización y edición de los documentos generados, optimizando su esfuerzo y asegurando el cumplimiento de los estándares establecidos.

## Hipótesis y Objetivos Iniciales

La **hipótesis** sobre la que trabajamos desde el principio fue que podríamos generar **resúmenes útiles** para los abogados, tales que les ahorren una cantidad considerable de tiempo, solo usando **prompt engineering**, sin la necesidad de **fine-tunear** algún modelo.



## Resultados Finales

Los resultados obtenidos con el modelo **Llama3.2-11B-Visual-Instruct** en configuración de 16 bits se resumen en las columnas **Metadatos** y **Síntesis** de la siguiente tabla:

![Desempeño del Modelo Llama3.2-11B-Visual-Instruct 16 bit](llama3_2_performance.png)

El modelo demostró un desempeño excelente en las tareas de extracción de metadatos y redacción de síntesis, alcanzando un 100% de buen formato y cohesión/coherencia. En la tarea de generación de sumarios, también obtuvo un desempeño destacado con un 96% de buen formato y cohesión/coherencia, aunque un 4% presentó problemas menores de formato y cohesión, sin embargo, estos resultados de generación de sumarios fue obtenida por fuera del trabajo realizado para la materia Mineria de Datos para el procesamiento de Texto. El tiempo total de procesamiento fue de 31 minutos.


## Planificación y Ejecución

El desarrollo de este proyecto implicó un proceso iterativo de aprendizaje y experimentación con diversos entornos y modelos hasta alcanzar la solución final. A continuación, se detalla esta trayectoria.

### Familiarización con los Modelos
En una primera instancia, se utilizó la plataforma **Google Colab** para comprender el manejo de modelos de lenguaje. Durante esta etapa inicial, el enfoque estuvo en:
- Cargar y utilizar modelos preentrenados.
- Experimentar con bibliotecas especializadas como **Transformers**, **Accelerate** y **Torch**.
Aprovechando los 15 GB de GPU de la versión gratuita de Colab, fue posible realizar pruebas básicas y obtener los primeros resultados.

### Acceso a Recursos de Alto Desempeño
En la segunda fase, se accedió a servidores avanzados en San Francisco, Córdoba, equipados con dos **NVIDIA A30** de 25 GB de GPU cada una. Este entorno permitió:
- Cargar un modelo **LLaMA de 8B parámetros**.
- Obtener resultados prometedores y eficientes en la extracción de metadatos y la generación de síntesis.

### Decisiones Estratégicas: Selección del Modelo
Tras un análisis exhaustivo de los pros y contras de varios modelos, se decidió trabajar con modelos de la familia **LLaMA**, destacando las siguientes decisiones:
- No utilizar modelos con tareas predefinidas, como **LLaMA3.1 8B Instruct-Summarization**, que inicialmente parecía una opción viable pero ofrecía menor flexibilidad.
- Priorizar modelos **Instruct-based**, los cuales permitieron una mayor adaptabilidad y personalización en la generación de texto.

Este enfoque iterativo y basado en decisiones estratégicas fue clave para lograr un balance entre desempeño, flexibilidad y recursos computacionales disponibles.


## Librerías y Codebases
- **[Transformers](https://huggingface.co/docs/transformers/)**: Biblioteca para trabajar con modelos de lenguaje preentrenados.
- **[Python](https://www.python.org/)**: Lenguaje de programación principal.
- **[CUDA](https://developer.nvidia.com/cuda-toolkit)**: Para la aceleración en GPU.
- **[Torch](https://pytorch.org/)**: Framework para machine learning.
- **[Modelo Llama-3.2-11B-Vision-Instruct](https://github.com/facebookresearch/llama)**: Modelo preentrenado especializado.

## Feedback e Integración de Sugerencias

- **Comentarios Recibidos:**  
  Uno de los puntos más destacados en el feedback de nuestros compañeros de materia fue la preocupación sobre cómo determinar la calidad de los resúmenes generados. Las preguntas principales giraban en torno a las métricas utilizadas, los métodos de evaluación y los criterios para definir si un resumen es efectivo.

- **Implementación de Sugerencias:**  
  Para abordar estas inquietudes, nos contactamos con la Dra. Valeria Sola, del Poder Judicial de Córdoba, con el objetivo de someter los resúmenes generados por nuestro modelo a una evaluación experta. Los resúmenes seran revisados por abogados especializados, quienes evaluaran su calidad en diferentes aspectos, como precisión, claridad y utilidad en el contexto judicial.


## Metodología

- **Preprocesamiento:**  
  Los textos judiciales originales fueron transformados a formato de texto plano para que pudieran ser procesados eficientemente por un modelo de lenguaje grande (LLM). Este paso incluyó la limpieza de datos, eliminación de formatos no compatibles y adaptación del contenido a una estructura adecuada para el modelo.

- **Entrenamiento/Inferencia:**  
  Para obtener resultados de alta calidad, se implementaron técnicas avanzadas como **Chain of Thought (CoT)** y **Few-Shot Encoding**. Cada sección del resumen fue abordada con prompts específicos, diseñados cuidadosamente para incluir ejemplos relevantes y específicos al contexto de esa parte del texto. Esto permitió al modelo comprender mejor la tarea y generar resúmenes más precisos y adaptados al ámbito jurídico.


## Resultados

Estamos a la espera del feedback de los abogados sobre los resúmenes generados, como se detalla en la sección **"Feedback e Integración de Sugerencias"**. Consideramos que esta evaluación experta es la métrica más relevante y confiable para valorar la calidad de nuestro trabajo, ya que proviene de profesionales del ámbito judicial.

Mientras tanto, los resultados actuales, presentados en la sección **"Resultados Finales"**, indican un desempeño prometedor. Estos resultados reflejan la capacidad del modelo para generar resúmenes coherentes y adaptados al contexto judicial, pero su validez definitiva dependerá del análisis y las recomendaciones de los expertos.


## Conclusiones y Trabajo Futuro

Este proyecto demostró la viabilidad de utilizar modelos de lenguaje grande (LLMs) para automatizar parcialmente la creación de sumarios de jurisprudencia, abordando tareas como la extracción de metadatos y la redacción de síntesis. Al aplicar técnicas de **Prompt Engineering** y estrategias como **Few-Shot Prompting** y cuantización, logramos resultados prometedores sin necesidad de realizar costosos ajustes de fine-tuning.

### Principales Contribuciones:
- Validación de técnicas como **Few-Shot Prompting**, obteniendo resultados consistentes en tareas complejas.
- Uso de la cuantización para manejar modelos grandes en hardware limitado, manteniendo una alta calidad de generación.
- Identificación del modelo **Llama3.2-11B-Visual-Instruct** como un balance óptimo entre calidad y tiempos de inferencia.

Sin embargo, también se identificaron desafíos, como problemas ocasionales de "alucinación" en textos complejos y limitaciones de contexto en documentos extensos. Estas áreas destacan la importancia de optimizar tanto los modelos como los prompts para dominios específicos.

### Trabajo Futuro:
- **Análisis del Feedback de abogados:** Incorporar las sugerencias de expertos para mejorar los resultados.
- **Integración en flujos de trabajo reales:** Implementar un piloto en el Boletín Judicial para evaluar su impacto práctico.
- **Ampliación del conjunto de datos:** Incluir nuevos fueros y jurisdicciones para probar la generalización del sistema.
- **Mejora de evaluaciones:** Desarrollar métricas automáticas y subjetivas para medir la calidad y utilidad de los resúmenes.
- **Exploración de modelos multimodales:** Investigar modelos que combinen texto e imágenes para procesar documentos legales escaneados.

## Requisitos del Proyecto

Asegúrate de contar con los siguientes requisitos antes de ejecutar el proyecto:

### Herramientas Necesarias
- **Python 3.8 o superior**: Lenguaje de programación principal.
- **CUDA 11.7 o superior**: Para aprovechar la aceleración en GPU (opcional si no se utiliza GPU).

### Dependencias
Instala las siguientes bibliotecas mediante `pip`:
```bash
pip install transformers torch tqdm
```

## Uso

Sigue los pasos a continuación para configurar y ejecutar el proyecto:

### Clonar el Repositorio
Clona este repositorio en tu máquina local y accede al directorio del proyecto:
```bash
git clone https://github.com/tuusuario/tu-repositorio.git
cd tu-repositorio
```

### Configurar el Modelo
El proyecto utiliza el modelo preentrenado **meta-llama/Llama-3.2-11B-Vision-Instruct**. Asegúrate de tener acceso al modelo desde Hugging Face en el siguiente enlace:  
- [Llama-3.2-11B-Vision-Instruct en Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

### Preparar los Datos
Coloca los textos legales en formato `.txt` dentro de la carpeta `data/input`. Asegúrate de que estén preprocesados correctamente para su análisis.

### Ejecutar el Script Principal
Para realizar las tareas de extracción de metadatos y generación de síntesis, ejecuta el siguiente comando:
```bash
python main.py
```

### Personalizar configuraciones (opcional)
Los parámetros configurables incluyen:
- **max_new_tokens**: Número máximo de tokens generados.
- **temperature**: Controla la diversidad de la generación.
- **min_new_tokens**: Número mínimo de tokens generados.

## Referencias

A continuación, se enumeran las principales fuentes y referencias bibliográficas utilizadas en este proyecto:

1. [Documentación de Transformers - Hugging Face](https://huggingface.co/docs/transformers/en/index): Recurso oficial para comprender el uso y funcionalidades de la biblioteca Transformers.
2. [Modelos LLaMA - Meta](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf): Colección oficial de modelos LLaMA en Hugging Face.
3. [Quantization en LLM - Tech Research Space](https://medium.com/@techresearchspace/what-is-quantization-in-llm-01ba61968a51): Explicación detallada sobre la cuantización y su impacto en modelos de lenguaje.
4. [Chain of Thought Prompting - Deepgram](https://deepgram.com/learn/chain-of-thought-prompting-guide): Guía sobre la técnica de Chain of Thought Prompting para mejorar la precisión de LLMs.
5. [Few-shot Prompting - Learn Prompting](https://learnprompting.org/docs/basics/few_shot?srsltid=AfmBOopY1hz1ELtbvzCDem35nexC4X8LwZImQlWRuWQ2f1k8YcqNk3u_): Introducción al uso de Few-shot Prompting para personalizar respuestas de modelos de lenguaje.


