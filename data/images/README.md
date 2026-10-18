# Imágenes para el Task 1

Las tres imágenes de entrada. Si se cambia algún nombre, actualice también la celda de configuración del notebook.

| Caso | Archivo | Contenido | Fuente | Licencia o permiso |
|---|---|---|---|---|
| (a) | `a_objeto_claro.jpg` | Pastor alemán sentado y centrado (clase 235 de ImageNet) | Magnific, "Dog is sitting, white background, with black brown face", foto premium ID 369849084: <https://www.magnific.com/premium-photo/dog-is-sitting-white-background-with-black-brown-face_369849084.htm> | Foto premium de Magnific, usada solo con fines académicos |
| (b) | `b_dos_objetos.jpg` | Pastor alemán y gato atigrado (clases 235 y 281) | Vecteezy, "Happy dog and cat that looking at the camera together isolated on white background", de Adiroot Komsawart, ID 47006768: <https://www.vecteezy.com/photo/47006768-happy-dog-and-cat-that-looking-at-the-camera-together-isolated-on-white-background> | Licencia gratuita de Vecteezy con atribución requerida |
| (c) | `c_contexto_enganoso.jpg` | Banana sobre una laptop junto a un mouse (clase 954) | Dreamstime, "Banana sobre la mesa cerca del ordenador con un ratón de refrigerio", ID 293587325: <https://es.dreamstime.com/banana-sobre-la-mesa-cerca-del-ordenador-con-un-rat%C3%B3n-de-refrigerio-image293587325> | Vista previa con marca de agua de Dreamstime, usada solo con fines académicos |

El recuadro de la imagen (c) es `(x_min, y_min, x_max, y_max) = (0, 80, 223, 208)`, en coordenadas de la imagen **después** de redimensionar el lado corto a 256 y recortar al centro a 224 × 224. El notebook lo usa como `[y_min:y_max, x_min:x_max]`, con el extremo exclusivo.
