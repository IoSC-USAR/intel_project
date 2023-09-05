import tensorflow as tf
import cv2
#loading model------------------------------
my_model = tf.keras.models.load_model('intel_model.h5')
#testing
test=tf.random.normal((1,224,224,3),seed=54)
test=test/255.
print(my_model.predict(test))
#--------------------------------------------
print("--")
#api


from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = my_model.predict(image)
        print("--------------------------",predictions)
        predicted_probability = float(predictions[0][0])

        if predicted_probability >= 0.5:
            predicted_class = "pneumonia"
        else:
            predicted_class = "normal"

        return {"class_name": predicted_class, "probability": predicted_probability}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
#uvicorn main:app --reload