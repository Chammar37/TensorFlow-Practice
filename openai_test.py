import os
import openai


# print(os.getenv('OPENAI_API_KEY'))
openai.api_key = "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"

response = openai.Completion.create(
  model="code-davinci-002",
  prompt="#Simple tensorflow code for image recognition  CNN\n\nimport tensorflow as tf\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport os\nimport cv2\nimport random\nimport pickle\n\nDATADIR = \"C:/Users/Siddharth/Desktop/Python/Tensorflow/PetImages\"\nCATEGORIES = [\"Dog\", \"Cat\"]\nIMG_SIZE = 50\n\ntraining_data = []\n\ndef create_training_data():\n    for category in CATEGORIES:\n        path = os.path.join(DATADIR, category)\n        class_num = CATEGORIES.index(category)\n        for img in os.listdir(path):\n            try:\n                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)\n                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))\n                training_data.append([new_array, class_num])\n            except Exception as e:\n                pass\n\ncreate_training_data()\n\nrandom.shuffle(training_data)\n\nX = []\ny = []\n\nfor features, label in training_data:\n    X.append(features)\n    y.append(label)\n\nX = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)\n\npickle_out = open(\"X.pickle\", \"wb\")\npickle.dump(X, pickle_out)\npickle_out.close()\n\npickle_out = open(\"y.pickle\", \"wb\")\npickle.dump(y, pickle_out)\npickle_out.close()\n\npickle_in = open(\"X.pickle\", \"rb\")\nX = pickle.load(pickle_in)\n\npickle_in = open(\"y.pickle\", \"rb\")\ny = pickle.load(pickle_in)\n\nX = X/255.0\n\nmodel = tf.keras.models.Sequential()\n\nmodel.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = X.shape[1:]))\nmodel.add(tf.keras.layers.Activation(\"relu\"))\nmodel.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))\n\nmodel.add(tf.keras.layers.Conv2D(64, (3,3)))\nmodel.add(tf.keras.layers.Activation(\"relu\"))\nmodel.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))\n\nmodel.add(tf.keras.layers.Flatten())\nmodel.add(tf.keras.layers.Dense(64))\n\nmodel.add(tf.keras.layers.Dense(1))\nmodel.add(tf.keras.layers.Activation(\"sigmoid\"))\n\nmodel.compile(loss = \"binary_crossentropy\",\n              optimizer = \"adam\",\n              metrics = [\"accuracy\"])\n\nmodel.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1)",
  temperature=0.5,
  max_tokens=300,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response)