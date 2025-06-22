# Modelos exportados
Como algunos modelos tardan en entrenar aprovechamos y exportamos los pesos con `torch.save` analogamente con `torch.load` lo recuperamos. Los files estan [aca](https://drive.google.com/drive/folders/1ilkeHheRd4tbJreB0pnW0c5JWYUi19RO?usp=sharing)

# Change Log

- 20/06 fabro

    1. separo los criterios de Loss en puntuacion y capitalizacion.
    2. defino los `weight =` para cada uno para ponderar el costo de error para las clases desbalanceadas. 
    3. des-freezeo las ultimas 2 capas de BERT y su pooler para "fine-tunear" a BERT
    4. guarde en un .pt los modelos pero son pesados para subirlos a git

- 21/06 fabro:

    1. separe el notebook en archivos de utilidad

- 21/06 male:

    1. agrego datasets de dialogos de peliculas
    2. modifico la funcion de evaluacion para incluir reporte de clasificacion (revisar por qué se rompe)

- 22/06 fabro:

    1. modifique un poco mas la estructura de carpetas
    2. cree un notebook para entrenar la red bidireccional
    3. agregue una fuente de datos que combina preguntas (de las que no usabamos) y las combina con afirmaciones
    4. probe congelar todo BERT salvo las ultimas 2 capas en lugar de descongelar las cosas que no debiamos
