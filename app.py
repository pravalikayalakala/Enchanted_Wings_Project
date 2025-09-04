from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

app = Flask(__name__)

# ========== Step 1: Download the model from Google Drive if not already there ==========
MODEL_PATH = 'vgg16_model.h5'
DRIVE_FILE_ID = '1M8buZI_li1f0PQ52LA7hGtK31tjIqc_v'  # ⬅️ Replace with your actual Google Drive file ID  
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")

# ========== Step 2: Load the model ==========
model = load_model(MODEL_PATH)

# ========== Step 3: Upload folder setup ==========
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ========== Step 4: Butterfly Class Names ==========
class_names = {
    0: 'Achalarus lyciades',
    1: 'Anartia jatrophae',
    2: 'Ancyloxypha numitor',
    3: 'Aphrissa statira',
    4: 'Atlides halesus',
    5: 'Battus philenor',
    6: 'Calycopis cecrops',
    7: 'Celastrina ladon',
    8: 'Chlosyne lacinia',
    9: 'Chlosyne nycteis',
    10: 'Colias eurytheme',
    11: 'Danaus gilippus',
    12: 'Danaus plexippus',
    13: 'Dryas iulia',
    14: 'Epargyreus clarus',
    15: 'Euptoieta claudia',
    16: 'Eurema lisa',
    17: 'Eurema nicippe',
    18: 'Eurytides marcellus',
    19: 'Hemiargus ceraunus',
    20: 'Junonia coenia',
    21: 'Limenitis arthemis',
    22: 'Lycaena phlaeas',
    23: 'Megisto cymela',
    24: 'Papilio cresphontes',
    25: 'Papilio glaucus',
    26: 'Papilio polyxenes',
    27: 'Pheos altatus',
    28: 'Phoebis philea',
    29: 'Phyciodes tharos',
    30: 'Pieris rapae',
    31: 'Pyrgus communis',
    32: 'Satyrium calanus',
    33: 'Strymon melinus',
    34: 'Vanessa atalanta',
    35: 'Vanessa cardui',
    36: 'Vanessa virginiensis',
    37: 'Zerene cesonia',
    38: 'Morpho peleides',
    39: 'Catopsilia pomona',
    40: 'Kallima inachus',
    41: 'Idea leuconoe',
    42: 'Troides aeacus',
    43: 'Graphium agamemnon',
    44: 'Papilio machaon',
    45: 'Delias eucharis',
    46: 'Appias albina',
    47: 'Pontia daplidice',
    48: 'Eurema hecabe',
    49: 'Colotis etrida',
    50: 'Pareronia valeria',
    51: 'Cepora nerissa',
    52: 'Spindasis vulcanus',
    53: 'Curetis thetis',
    54: 'Jamides celeno',
    55: 'Anthene emolus',
    56: 'Lethe europa',
    57: 'Ypthima baldus',
    58: 'Mycalesis mineus',
    59: 'Junonia almana',
    60: 'Hypolimnas bolina',
    61: 'Ariadne merione',
    62: 'Acraea violae',
    63: 'Danaus chrysippus',
    64: 'Tirumala limniace',
    65: 'Euploea core',
    66: 'Papilio demoleus',
    67: 'Graphium doson',
    68: 'Papilio memnon',
    69: 'Papilio polymnestor',
    70: 'Pachliopta aristolochiae',
    71: 'Troides helena',
    72: 'Cethosia cyane',
    73: 'Vindula erota',
    74: 'Parthenos sylvia'
}

# ========== Step 5: Routes ==========
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected!', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    predicted_label = class_names.get(predicted_class, "Unknown Species")

    return render_template('output.html', prediction=predicted_label, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

