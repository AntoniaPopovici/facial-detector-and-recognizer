from imports import *
from FacialRecognizer import *
from generator import *
from utils import *
from evals import *
from params import *
import uuid

os.makedirs(C_FACES, exist_ok=True)
os.makedirs(BARNEY, exist_ok=True)
os.makedirs(NON_BARNEY, exist_ok=True)
os.makedirs(BETTY, exist_ok=True)
os.makedirs(NON_BETTY, exist_ok=True)
os.makedirs(FRED, exist_ok=True)
os.makedirs(NON_FRED, exist_ok=True)
os.makedirs(WILMA, exist_ok=True)
os.makedirs(NON_WILMA, exist_ok=True)
os.makedirs(FINAL_RESULTS_TASK2, exist_ok=True)


# os.makedirs('descriptori_barney', exist_ok=True)

# CHARACTERS = ['barney', 'betty', 'fred', 'wilma']
# CHARACTER_FOLDERS = {'barney': 'antrenare/barney',
#                      'betty': 'antrenare/betty',
#                      'fred': 'antrenare/fred',
#                      'wilma': 'antrenare/wilma'}
# ANNOTATION_FILES = {'barney': 'antrenare/barney_annotations.txt',
#                     'betty': 'antrenare/betty_annotations.txt',
#                     'fred': 'antrenare/fred_annotations.txt',
#                     'wilma': 'antrenare/wilma_annotations.txt'}
# OUTPUT_FOLDERS = {'barney': (BARNEY, NON_BARNEY),
#                   'betty': (BETTY, NON_BETTY),
#                   'fred': (FRED, NON_FRED),
#                   'wilma': (WILMA, NON_WILMA)}


# output_negative_faces_folder = 'exemple_caractere/imaginiNeg'
# os.makedirs(output_negative_faces_folder, exist_ok=True)


# for character in CHARACTERS:
#     for character2 in CHARACTERS:
#         if character != character2:
#             character_annotations_file = ANNOTATION_FILES[character]
#             character_annotations_file2 = ANNOTATION_FILES[character2]
#             character_folder = CHARACTER_FOLDERS[character]
#             character_folder2 = CHARACTER_FOLDERS[character2]

#             output_positive_folder, output_negative_folder = OUTPUT_FOLDERS[character]
#             output_positive_folder2, output_negative_folder2 = OUTPUT_FOLDERS[character2]

#             generate_positive_faces_per_character(
#                 character_annotations_file, character_annotations_file2, character_folder, character_folder2, output_positive_folder, output_positive_folder2, character, character2, output_negative_folder, output_negative_folder2)

#     generate_negative_non_faces(
#         ANNOTATION_FILES[character], CHARACTER_FOLDERS[character], output_negative_faces_folder)


# def copy_images(source_folder, destination_folder, character_name, second_character):
#     os.makedirs(destination_folder, exist_ok=True)

#     for root, _, files in os.walk(source_folder):
#         for file in files:
#             source_path = os.path.join(root, file)

#             unique_identifier = str(uuid.uuid4())[:8]

#             title, ext = os.path.splitext(file)
#             new_title = f'{title}_{character_name}_{second_character}_{unique_identifier}{ext}'

#             destination_path = os.path.join(destination_folder, new_title)
#             shutil.copy(source_path, destination_path)


# os.makedirs('exemple_caractere/barney/imaginiPozitive', exist_ok=True)
# os.makedirs('exemple_caractere/betty/imaginiPozitive', exist_ok=True)
# os.makedirs('exemple_caractere/fred/imaginiPozitive', exist_ok=True)
# os.makedirs('exemple_caractere/wilma/imaginiPozitive', exist_ok=True)
# os.makedirs('exemple_caractere/barney/imaginiNegative', exist_ok=True)
# os.makedirs('exemple_caractere/betty/imaginiNegative', exist_ok=True)
# os.makedirs('exemple_caractere/fred/imaginiNegative', exist_ok=True)
# os.makedirs('exemple_caractere/wilma/imaginiNegative', exist_ok=True)


# positive_images_folder_barney = os.path.join(
#     'exemple_caractere/barney', 'imaginiPozitive')
# positive_images_folder_betty = os.path.join(
#     'exemple_caractere/betty', 'imaginiPozitive')
# positive_images_folder_fred = os.path.join(
#     'exemple_caractere/fred', 'imaginiPozitive')
# positive_images_folder_wilma = os.path.join(
#     'exemple_caractere/wilma', 'imaginiPozitive')


# negative_images_folder_barney = os.path.join(
#     'exemple_caractere/barney', 'imaginiNegative')
# negative_images_folder_betty = os.path.join(
#     'exemple_caractere/betty', 'imaginiNegative')
# negative_images_folder_fred = os.path.join(
#     'exemple_caractere/fred', 'imaginiNegative')
# negative_images_folder_wilma = os.path.join(
#     'exemple_caractere/fred', 'imaginiNegative')


# copy_images('fete_caractere/barney_poz',
#             positive_images_folder_barney, 'barney', 'barney')
# copy_images('fete_caractere/barney_neg',
#             negative_images_folder_barney, 'barney', 'barney')
# copy_images('fete_caractere/betty_pos',
#             negative_images_folder_barney, 'barney', 'betty')
# copy_images('fete_caractere/fred_pos',
#             negative_images_folder_barney, 'barney', 'fred')
# copy_images('fete_caractere/wilma_pos',
#             negative_images_folder_barney, 'barney', 'wilma')
# copy_images('exemple_caractere/imaginiNeg',
#             negative_images_folder_barney, 'barney', 'img_neg')


# copy_images('fete_caractere/betty_poz',
#             positive_images_folder_betty, 'betty', 'betty')
# copy_images('fete_caractere/betty_neg',
#             negative_images_folder_betty, 'betty', 'betty')
# copy_images('fete_caractere/barney_poz',
#             negative_images_folder_betty, 'betty', 'barney')
# copy_images('fete_caractere/fred_poz',
#             negative_images_folder_betty, 'betty', 'fred')
# copy_images('fete_caractere/wilma_poz',
#             negative_images_folder_betty, 'betty', 'wilma')
# copy_images('exemple_caractere/imaginiNeg',
#             negative_images_folder_betty, 'betty', 'img_neg')

# copy_images('fete_caractere/fred_poz',
#             positive_images_folder_fred, 'fred', 'betty')
# copy_images('fete_caractere/fred_neg',
#             negative_images_folder_fred, 'fred', 'fred')
# copy_images('fete_caractere/barney_poz',
#             negative_images_folder_fred, 'fred', 'barney')
# copy_images('fete_caractere/betty_poz',
#             negative_images_folder_fred, 'fred', 'betty')
# copy_images('fete_caractere/wilma_poz',
#             negative_images_folder_fred, 'fred', 'wilma')
# copy_images('exemple_caractere/imaginiNeg',
#             negative_images_folder_fred, 'fred', 'img_neg')

# copy_images('fete_caractere/wilma_poz',
#             positive_images_folder_wilma, 'wilma', 'wilma')
# copy_images('fete_caractere/wilma_neg',
#             negative_images_folder_wilma, 'wilma', 'wilma')
# copy_images('fete_caractere/barney_poz',
#             negative_images_folder_wilma, 'wilma', 'barney')
# copy_images('fete_caractere/fred_poz',
#             negative_images_folder_wilma, 'wilma', 'fred')
# copy_images('fete_caractere/betty_poz',
#             negative_images_folder_wilma, 'wilma', 'betty')
# copy_images('exemple_caractere/imaginiNeg',
#             negative_images_folder_wilma, 'wilma', 'img_neg')


facial_recognizer = FacialRecognizer()


positive_features_path = os.path.join(
    'descriptori_barney', 'descriptori_pozitivi_barney.npy')
if os.path.exists(positive_features_path):
    positive_descriptors_barney = np.load(positive_features_path)
    print(positive_descriptors_barney.shape)
    print('Descriptori pozitivi incarcati')
else:
    print('Se genereaza descriptori pozitivi...')
    positive_descriptors_barney = facial_recognizer.get_positive_descriptors_characters(
        path='exemple_caractere/barney/imaginiPozitive')
    print(positive_descriptors_barney.shape)
    np.save(positive_features_path, positive_descriptors_barney)

negative_features_path = os.path.join(
    'descriptori_barney', 'descriptori_negativi_barney.npy')
if os.path.exists(negative_features_path):
    negative_descriptors_barney = np.load(negative_features_path)
    print(negative_descriptors_barney.shape)
    print('Descriptori negativi incarcati')
else:
    print('Se genereaza descriptori negativi...')
    negative_descriptors_barney = facial_recognizer.get_negative_descriptors_characters(
        path='exemple_caractere/barney/imaginiNegative')
    print(negative_descriptors_barney.shape)
    np.save(negative_features_path, negative_descriptors_barney)


training_examples = np.concatenate(
    (np.squeeze(positive_descriptors_barney), np.squeeze(negative_descriptors_barney)), axis=0)
print("Numarul de caracteristici folosite in modelul de antrenare:",
      training_examples.shape[1])

train_labels = np.concatenate(
    (np.ones(positive_descriptors_barney.shape[0]), np.zeros(negative_descriptors_barney.shape[0])))
facial_recognizer.train_classifier(training_examples, train_labels)


detections, scores, file_names = facial_recognizer.run()
# eval_detections('validare/task2_barney_gt_validare.txt',
#                'rezultat', detections, scores, file_names)
# show_detections_with_ground_truth(detections, scores, file_names)

np.save(FINAL_RESULTS_TASK2 + 'detections_all_faces_barney.npy', detections)
np.save(FINAL_RESULTS_TASK2 + 'scores_all_faces_barney.npy', scores)
np.save(FINAL_RESULTS_TASK2 + 'file_names_all_faces_barney.npy', file_names)
