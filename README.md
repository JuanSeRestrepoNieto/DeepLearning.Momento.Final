# Faster R-CNN: detección de peatones con Region Proposal Networks

Proyecto final del curso **Introducción al aprendizaje profundo** (Institución Universitaria ITM), *Momento Evaluativo: Aplicaciones del aprendizaje profundo*.

- **Aplicación:** detección de objetos (Anexo A, punto 5)
- **Artículo:** Ren, He, Girshick y Sun (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) (copia en [Docs/1506.01497v3.pdf](Docs/1506.01497v3.pdf))
- **Dataset:** [Penn-Fudan Pedestrian](https://www.cis.upenn.edu/~jshi/ped_html/): 170 imágenes, 423 peatones con máscara de instancia
- **Métricas (Anexo A):** Precision, Recall, IoU y mAP

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanSeRestrepoNieto/DeepLearning.Momento.Final/blob/main/notebooks/faster_rcnn_penn_fudan.ipynb)

## Estructura

```
Docs/                                  Enunciado, Anexo A, paper, diapositivas del curso y Parte 1 (fundamentación)
notebooks/faster_rcnn_penn_fudan.ipynb Implementación completa (entregable de código)
results/                               Figuras y tablas que genera el notebook (para las diapositivas de la Parte 2)
requirements.txt                       Dependencias (Colab ya las trae)
```

## Cómo ejecutar

**Google Colab (recomendado):** abra el notebook con el botón de arriba y elija `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`. Luego ejecute `Entorno de ejecución → Ejecutar todas`. Tarda unos 25–35 minutos y descarga el dataset y los pesos preentrenados automáticamente. Al terminar, descargue la carpeta `results/` (panel de archivos de Colab).

**Local:**

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook notebooks/faster_rcnn_penn_fudan.ipynb
```

Para comprobar el flujo completo en pocos minutos, incluso en CPU, active el modo rápido (pocas imágenes, 1 época, imágenes de 320 px). Este modo solo sirve para probar el código; sus números no son válidos como resultados:

```bash
QUICK_RUN=1 jupyter nbconvert --to notebook --execute notebooks/faster_rcnn_penn_fudan.ipynb --output /tmp/prueba.ipynb
```

## Qué contiene el notebook

| Sección | Contenido | Relación con la Parte 1 (diapositivas) |
|---|---|---|
| 2. Datos | Descarga, `Dataset` de PyTorch, partición train/val/test fija (100/20/50), EDA de tamaños y relaciones de aspecto frente a las anclas | Anclas: escalas y relaciones de aspecto |
| 3. Faster R-CNN por dentro | Parámetros por componente, recorrido paso a paso del *forward*, dimensiones `cls` (k) / `reg` (4k), anclas positivas, negativas e ignoradas, codificación $(t_x,t_y,t_w,t_h)$, mapa de *objectness* y top-N propuestas | Arquitectura unificada, anatomía de la RPN, dimensiones matemáticas, pérdida multitarea |
| 4. Métricas | IoU, Precision, Recall, F1, AP@0.5, AP@0.75, mAP@[.5:.95] y taxonomía de errores (localización, duplicado, fondo, FN), con pruebas unitarias | — |
| 5. Experimentos | **E0** zero-shot COCO · **E1** extractor congelado · **E2** fine-tuning · **E3** MobileNetV3 · **E4** entrenamiento alterno de 4 pasos | Entrenamiento en 4 pasos; *trade-off* ZF vs. VGG |
| 6. Resultados | Curvas de pérdida por componente, tabla comparativa, curvas PR, AP según el IoU, barrido del umbral | — |
| 7. Cualitativo | Predicciones comparadas, distribución de errores, histograma de IoU, galería de errores | Falsos positivos y errores de localización |
| 8. Complejidad | Tiempo por componente (backbone / RPN / cabeza) y ablación del número de propuestas (10 a 1000) | Análisis de tiempos de inferencia |
| 9. Conclusiones | Resumen automático (`results/resumen.json`) y guía de discusión | — |

## Decisiones de implementación y diferencias con el paper

- Se usa la implementación de **torchvision** (`fasterrcnn_resnet50_fpn`), preentrenada en COCO, como permite el enunciado. A diferencia del paper (VGG-16 o ZF, un solo mapa de características, 9 anclas), usa ResNet-50 con **FPN**: 5 niveles con un tamaño de ancla por nivel y 3 relaciones de aspecto. La RPN, las anclas, la pérdida multitarea y el detector Fast R-CNN siguen el diseño del artículo.
- La capa `cls` de la RPN produce **k** salidas con sigmoide en lugar de **2k** con softmax. Para dos clases son equivalentes, y el propio paper lo menciona.
- El entrenamiento por defecto es el **conjunto aproximado** (Sec. 3.2-ii del paper). El **entrenamiento alterno de 4 pasos** se implementa aparte en E4 para compararlos.
- Penn-Fudan no tiene partición oficial. Se fija una con semilla 42 y el conjunto de prueba nunca se usa para tomar decisiones: el mejor modelo y el umbral de confianza se eligen en validación.

## Referencias

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN. *NeurIPS*.
- Lin, T.-Y. et al. (2017). Feature Pyramid Networks for Object Detection. *CVPR*.
- Howard, A. et al. (2019). Searching for MobileNetV3. *ICCV*.
- Wang, L., Shi, J., Song, G., & Shen, I. (2007). Object Detection Combining Recognition and Segmentation. *ACCV* (Penn-Fudan).
- Hoiem, D., Chodpathumwan, Y., & Dai, Q. (2012). Diagnosing Error in Object Detectors. *ECCV*.
- [Tutorial oficial de detección de PyTorch](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
