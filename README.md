# MovieReviewPlatform

В этот репозиторий я скинул всё, что у меня получилось в ходе работы над сайтом для анализа отзывов на фильмы (только на английском!). Увидеть модель в действии можно вот по этой ссылке, если она ещё не отвалилась.

### Качество моделей

### Что лежит в репозитории

По убованию важности

1. `ml_training.ipynb`. Здесь находится всё, чем я занимался с классическими моделями машинного обучения: чуть-чуть предобработки, немножко EDA, выбор и обучения модели, тюнинг и интерпретирация модели. Там же есть интерфейс, вроде бы вполне рабочий, комментарии вроде бы достаточно подробные. Есть ещё смысл посмотреть эту же тетрадку [на кеггле](https://www.kaggle.com/code/yaustal/test-task-rosatom/notebook), там уже подключено всё, что нужно для работы
2. `dl_training.ipynb`. Тут то же самое, но уже для нейросеток. По большей части оказалось бесполезно, потому что на PythonAnywhere (а больше платформ нет, все отказались работать с русичами) ограничение в 500Мб, столько весит и торч, и моя моделька. Тем не менее, качество там получилось побить, я бы использовал именно их, если бы мог. Опять же удобнее всего глянуть [на кеггле]()
3. `report.pdf`. Там лежит отчёт о проделанной работе. Но он довольно краток, для подробных описаний надо смотреть в ноутбуки, если есть силы и желание
4. `logreg_0.905`. Это модель, которая пошла на сайт, её вид и точность даже прописаны в названии
5. `catboost_tuning.ipynb`. Тут я тюнил катбуст, но не доучил. У него был потенциал стать лучшей моделью, веса не слишком тяжёлые, но учится очень долго и требует гпу

Все ноутбуки скорее всего не прогружают графики в `plotly`, ещё один повод почекать их на кеггле

### Прочее

- [Тут](https://kaggle.com/datasets/72db9336f0d4d66b8187e1a072b315876c321e39837976fd538316f627b0feb8) лежат датасеты, которые я использовал в работе. Собственно данные, немного статистики и первичные метрики по моделям       
- [Тут](https://kaggle.com/datasets/72db9336f0d4d66b8187e1a072b315876c321e39837976fd538316f627b0feb8) лежат модели, которые я хотел использовать на сайте      
- [Здесь](https://wandb.ai/lerostre/IMDB_review_sentiment_analysis?workspace=user-lerostre) собраны метрики по нейросетям, очень красиво      
- [Это](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) статья, которую советовали к прочтению перед заданием. Единственное, что там интересного это метрики - их достигнутый максимум это около 0.88
- [Тут](https://paperswithcode.com/sota/sentiment-analysis-on-imdb) бенчмарки и интересные идели по применению нейросетей к этому датасету, здесь я черпал идеи для DL-ного ноутбука
