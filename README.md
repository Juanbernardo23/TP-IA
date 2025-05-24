## Implementación de Modelos de Lenguaje de Gran Tamaño (LLM) en la resolución de problemas matemáticos

Primer Fecha de presentación
24/5/2025

_Juan Antonio Bernardo_


# Qué es un LLM

Un modelo de lenguaje de gran tamaño (LLM, “Large Language Model”) es un tipo de red neuronal profunda entrenada sobre enormes cantidades de texto para “aprender” patrones estadísticos del lenguaje natural. A grandes rasgos, estos modelos se fundamentan en la arquitectura Transformer. La particularidad de los LLM consiste en su tamaño: suelen tener miles de millones (o incluso centenas de miles de millones) de parámetros entrenables, lo que les permite capturar relaciones complejas entre palabras, oraciones y documentos completos.
Aplicaciones de LLM en educación y en resolución de problemas matemáticos
Los LLM han abierto nuevas posibilidades en el ámbito educativo, tanto para docentes como para estudiantes. A continuación se describen algunas de las aplicaciones más relevantes:
Asistentes de estudio personalizados: Un LLM puede actuar como tutor virtual, contestando preguntas de manera inmediata y adaptando la explicación según el nivel de comprensión del alumno.
Generación de ejercicios y retroalimentación automática: Los LLM pueden generar conjuntos de problemas de práctica con variaciones en la dificultad y el formato de los enunciados. Asimismo, ante un enunciado resuelto por el alumno, el LLM puede revisar la solución y señalar errores en el razonamiento simbólico o aritmético, indicando en qué paso se produjo la falla.

Situación particular e inicial

## Hardware disponible:

__IMAGEN 1__

Como análisis inicial, mi hardware personal es decente para un entrenamiento ligero y ajustes mínimos de un LLM, especialmente sin una GPU dedicada. Aun así, luego de investigar e introducirnos en tema con técnicas como LoRA + CPU-friendly training y buena planificación, se puede lograr un entrenamiento funcional, aunque más lento y limitado.



# LoRA

LoRA (Low-Rank Adaptation) es una técnica moderna de fine-tuning para modelos grandes como LLMs (Large Language Models), que permite ajustar modelos enormes usando pocos recursos de hardware, como una PC sin GPU potente.

## ¿Qué hace LoRA?
En lugar de ajustar todos los millones o billones de parámetros del modelo original (lo cual es muy costoso en memoria y tiempo), LoRA:

Congela el modelo original.
Agrega capas pequeñas y livianas (matrices de bajo rango) dentro del modelo.
Solo entrena esas capas nuevas, manteniendo intacto el modelo base.

Requisitos para usar LoRA

Se necesitan las siguientes librerías en nuestra PC, como mínimo, al momento de hacer fine tuning:

transformers
peft
accelerate
datasets
bitsandbytes


Recomendaciones adicionales

Además, se nos recomendó inicialmente usar un tamaño de modelo pequeño, usar batch size = 1 y grad_accum para reducir el uso de memoria RAM dramáticamente al entrenar, a su vez, entrenar por tramos y guardar checkpoints y evitar tokenizaciones complejas.

Mi GPU integrada Radeon no soporta entrenamiento con frameworks como PyTorch + CUDA, ni siquiera a través de ROCm (soporte aún limitado).

Primeros pasos

Como primera decisión, elegí e instalé LLM Studio porque el programa me pareció bastante amigable para empezar a experimentar con un LM. 

Mi primera opción en cuanto a modelo, según la recomendación del programa, fue Qwen2.5-Math-7B, IA que según internet es de lenguaje de 7 mil millones de parámetros optimizado para razonamiento matemático. Forma parte de la serie Qwen2.5 desarrollada por Alibaba y destaca por su capacidad para resolver problemas matemáticos con múltiples pasos, incluyendo álgebra, aritmética y lógica simbólica. Es compatible con formatos como LaTeX y funciona bien en tareas que requieren precisión y seguimiento detallado de instrucciones matemáticas.

El modelo fue probado desde el mismo LLM Studio, haciendo una consulta breve en español, “¿cuánto es la raíz cuadrada de 81?” la cual generó una algo tardía respuesta, usando bastante RAM y obtuve una respuesta muy larga aunque acertada, pero completamente en inglés. 

__IMAGEN 2__

A partir de eso, se tomó la decisión de seguir viendo otras opciones para luego tratar de hacer un fine tuning en español orientado a lo visto en la materia, además de que buscando en internet, pude llegar a la conclusión de que la Qwen2.5-MAth-7B era un modelo bastante pesado para mi PC, (16 GB de Ram, un procesador bastante bueno pero una GPU integrada de 496 MB nomás) así que ese iba a ser el gran limitante, además del idioma ya que la idea es hacer prompts en español a largo plazo. Me instruí sobre cómo era la estructura y la forma de instalar otra IA localmente para su posterior intento de fine tuning, preferente usando VSC, en lo que pude aprender a cada escala que:

📄 train.py
Script principal que realiza el fine-tuning del modelo.
Carga los datos de entrenamiento, define la función de pérdida y guarda el modelo ajustado localmente.

📄 data/train.json
Archivo de entrenamiento en formato JSON, con una entrada por línea.
La idea en éste caso, fue grupalmente generar un JSON con varios ejercicios matemáticos, y para la segunda instancia de la investigación lograr un entrenamiento mucho más avanzado.
El JSON cuenta con 350 líneas. El contenido incluye ejercicios matemáticos de dificultad progresiva, desde temas introductorios hasta problemas más complejos. Cada entrada está estructurada como un intercambio tipo chat, con un mensaje del usuario y una respuesta del asistente. Hacia el final del conjunto se encuentran ejemplos avanzados, como desigualdades con valor absoluto y demostraciones algebraicas, lo que permite al modelo captar tanto el razonamiento simbólico como la claridad explicativa esperada.



Ej fragmento del JSON utilizado:

__IMAGEN 3__

📄 config.json
Archivo opcional para almacenar parámetros como nombre del modelo, número de épocas o rutas de archivos.
Permite centralizar la configuración del entrenamiento para facilitar ajustes futuros.

📄 chat.py
Script para interactuar con el modelo fine-tuneado en forma de chat local.
Recibe una entrada del usuario y devuelve la respuesta más semánticamente cercana entre una lista predefinida.

## paraphrase multilingual minilm l12 v2

Luego de seguir buscando otros modelos de IA convenientes para mi PC, idioma y al menos ahora un “entrenamiento simple”, me convencí de probar paraphrase multilingual minilm l12 v2 diseñada según las fuentes para generar representaciones vectoriales densas de 384 dimensiones a partir de oraciones o párrafos en más de 50 idiomas. Estas representaciones son útiles para tareas como búsqueda semántica, agrupamiento de textos y comparación de similitud entre frases. En éste caso, con algo ya más de idea sobre cómo debía proceder y observando que éste modelo pesaba alrededor de 1,5 GB, traté de directamente armar la estructura “modelo” para tratar de entrenarla localmente, ayudándome con ChatGPT para algunos ajustes, y que la misma vaya tomando forma, pude terminar el entrenamiento luego de muchos intentos durante varios días donde me fue dando muchos errores relacionados al CPU, al modelo en sí y a los comandos técnicos en el código. Aquí algunas capturas de pantalla de lo que fue el proceso, probando con varios ajustes en el train.py de distintos comandos, como:


__IMAGEN 4__


Captura del entrenamiento realizándose después de varios errores en el proceso:

__IMAGEN 5__

El entrenamiento finalmente tardó alrededor de dos horas y media una vez que pude dar con las configuraciones aparentemente correctas, tomando bien el JSON, y luego de ésto llegó el momento de probar el chat fine-tuneado mediante el script del chat.py, donde terminé viendo la primera salida: 

__IMAGEN 6__

Ésto fue muy interesante, porque además de detectar que la codificación estaba siendo muy extraña, la respuesta tardó alrededor de 5 minutos, utilizando toda la CPU y RAM. 

__IMAGEN 7__

Finalmente, después de muchos intentos fallidos y de buscar un poco más de información sobre el modelo, encontré que en realidad:
“No está entrenado para matemáticas: su arquitectura no entiende símbolos matemáticos ni realiza razonamiento formal como lo harían otros modelos.

## Phi-2

Por lo que se tuvo que tomar la decisión de dejar de perder tiempo y buscar otro candidato, el cual terminó siendo Phi-2, de Microsoft, un modelo de 2.7 mil millones de parámetros (2.7B), y que viene con un fuerte enfoque en calidad y eficiencia, ideal para tareas de lenguaje general, razonamiento y código ligero. También según investigué, es ideal para aplicar fine-tuning.

La estructura de las carpetas y archivos, luego de descargar localmente la IA y proponer el entrenamiento, finalmente fue:

mi-proyecto-finetuning/

├── data/

│   └── train.json                     # Dataset de entrenamiento con conceptos matemáticos

├── phi-2/                                # Modelo original sin fine-tuning

│   └── (...)                              # Archivos del modelo base Phi-2

├── phi-2-finetuned/                # Modelo ya ajustado

│   ├── added_tokens.json

│   ├── config.json

│   ├── generation_config.json

│   ├── merges.txt

│   ├── model.safetensors.index

│   ├── model-00001-of-00003.safetensors

│   ├── model-00002-of-00003.safetensors

│   ├── model-00003-of-00003.safetensors

│   ├── special_tokens_map.json

│   ├── tokenizer/

│   ├── tokenizer_config.json

│   ├── training_args.bin

│   └── vocab.json

├── checkpoints/                    # Checkpoints intermedios del fine-tuning

│   └── checkpoint-*/               # Subcarpetas con estados intermedios

├── config.py                       # Configuración general del proyecto

├── datamodule.py                  # Lógica de carga y procesamiento del dataset

├── train.py                        # Script principal de entrenamiento

├── chat.py                         # Script para cargar el modelo finetuneado y responder preguntas



El proceso de los entrenamientos fueron largos, pasando por muchos intentos con fallas las cuales tardaban muchas horas, luego al probarlas daban errores de codificación, ya que luego de analizar con detenimiento y generar alertas vía código, identificaba que el entrenamiento creaba muchos tokens vacíos en el tokenizer.json de salida (archivos muy largos), pero a eso solo lo podía comprobar entrenando el modelo y esperando, además de intentar adaptar el script de chat, quien también podría estar dando un mal resultado:

__IMAGEN 8__

Se estuvieron varios días para la búsqueda de entrenamiento correcto, hasta que por fin luego de adaptar algunos parámetros, se pudo crear una buena tokenización. El entrenamiento, a diferencia de los anteriores, tardó una hora más (aproximadamente 3 hs y media, los anteriores fallidos tardaban 2 hs y media, que no era poco) utilizando toda la RAM y el CPU en todos los casos:

__IMAGEN 9__

Una vez que se identificó vía chat.py que el entrenamiento fue correcto y que sólo faltaba adaptar el script del chat, comencé a ver resultados, como un prompt ingresado muy básico para probar la respuesta, y una demora de 2 minutos sumado a una gran utilización de RAM para finalmente dar con una alucinación inicial:

__IMAGEN 10__

Luego, en algunos otros casos se buscó una comprobación, dando una respuesta con otros números y mezclando una palabra en inglés con otra en español

__IMAGEN 11__

Luego de unos ajustes, se hicieron pruebas quizás un poco más complejas, dando una respuesta muy larga y un poco ilógica en un caso:

__IMAGEN 12__

Pero después de modificaciones, comenzó a dar respuestas correctas, aunque en un primer momento en algunas oportunidades seguía respondiendo con palabras en inglés, aunque con resultados correctos. Además de que pude verificar que el código ascii de la raíz cuadrada, por ejemplo, lo tomaba bien.

__IMAGEN 13__

Finalmente, se pudieron corregir las estrategias de respuesta, tokenización, condiciones para tomar el prompt, etc, y las respuestas fueron lógicas en la gran mayoría de los casos:

__IMAGEN 14__

__IMAGEN 15__

En los prompts más complejos, el tiempo de respuesta ha llegado hasta los 4 minutos, y en varios casos, a pesar de que en el JSON teníamos bastante entrenamiento preciso para casos complejos, no terminó de afianzarse lógica en algunas respuestas:

__IMAGEN 16__

