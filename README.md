## Init locally (not Docker)
1. If the Makefile is supposed to be used, then **make** itself must be installed
<br></br>

2. Packet initialization. About +-5GB, because torch+cuda are fat guys
<br></br>
   - If you have poetry installed (and make):
      ```bash
      make project-init
      ```
   - Just through pip (Don't forget to activate the virtual env!):

     ```bash
     pip install --no-cache-dir -r requirements.txt
     ```

## Deploy in Docker
1. Create an image. About +-5GB!
<br></br>
   - An image named **car-detection** will be created:
      ```bash
      make docker-make-image
      ```

2. Start container from image
<br></br>
    - With GPU:
      ```bash
      make docker-cont-gpu
      ```

    - With CPU:
      ```bash
      make docker-cont-cpu
      ```

Будет запущен *main.py* и можно будет перейти на html-страницу, где можно загрузить изображения и сделать предсказания \
Переменные окружения находятся в файле *.env*
<br></br>

### Check api work
Маршрут */api/pred*, метод *POST* \
На вход ожидается список изображений в формате base64-str \
Для каждого изображения вернется следующее: {"pred_image": annotated_img, "bboxes": bboxes}, где annotated_img и bboxes это списки \
\
Можно запустить файл **api_test**, который прочитает изображения в каталоге, переведет в формат base64-str, сделает POST-запрос, \
результаты переведет в np.array и выведет в консоль изображения и bounding боксы 

> **Note**: В каталоге **./car-detection/data/demo_data/** должны лежать несколько изображений для запуска *api_test.py* !

```bash
    python api_test.py
```


### Check cuda support
   ```bash
   python car_detection/cuda_itils.py
   ```

Была выбрана Yolo модель, потому что это быстро да и данных для обучения мало \
Качество можно было бы повысить следующими способами:
1) Разметить больше данных для обучения. Из предоставленной тысячи изображений размечена только треть, на них модель и обучена
2) Сделать аугментацию данных
3) Использовать более тяжелую модель, но для нее и данных нужно больше
4) Использовать другую модель, например, Faster RCNN
5) Поиграться с гиперпараметрами текущей модели
