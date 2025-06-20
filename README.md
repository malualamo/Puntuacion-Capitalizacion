# Modelos exportados
Como algunos modelos tardan en entrenar aprovechamos y exportamos los pesos con `torch.save` analogamente con `torch.load` lo recuperamos. Los files estan [aca](https://drive.google.com/drive/folders/1ilkeHheRd4tbJreB0pnW0c5JWYUi19RO?usp=sharing)

# Change Log

- 20/06 fabro

    1. separo los criterios de Loss en puntuacion y capitalizacion.
    2. defino los `weight =` para cada uno para ponderar el costo de error para las clases desbalanceadas. 
    3. des-freezeo las ultimas 2 capas de BERT y su pooler para "fine-tunear" a BERT
    4. guarde en un .pt los modelos pero son pesados para subirlos a git