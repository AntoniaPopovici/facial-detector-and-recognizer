from imports import *
from FacialDetector import *
from generator import *
from utils import *
from evals import *
from params import *

os.makedirs(POSITIVE, exist_ok=True)
os.makedirs(NEGATIVE, exist_ok=True)
os.makedirs(FACES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
os.makedirs('descriptori', exist_ok=True)
os.makedirs('rezultate', exist_ok=True)
os.makedirs(FINAL_RESULTS_TASK1, exist_ok=True)


# for character in ['barney', 'betty', 'fred', 'wilma']:
#     character_annotations_file = os.path.join(
#         TRAIN, f'{character}_annotations.txt')
#     character_folder = os.path.join(TRAIN, character)

#     output_negative_faces_folder = os.path.join(
#         FACES, f'{character}_non_fete')
#     output_positive_folder = os.path.join(
#         FACES, f'{character}_imagini_pozitive')
#     output_negative_folder = os.path.join(
#         FACES, f'{character}_imagini_negative')

#     os.makedirs(output_positive_folder, exist_ok=True)
#     os.makedirs(output_negative_folder, exist_ok=True)
#     os.makedirs(output_negative_faces_folder, exist_ok=True)

#     generate_positive_faces(character_annotations_file,
#                             character_folder, output_positive_folder)
#     generate_negative_faces(character_annotations_file,
#                             character_folder, output_negative_folder)
#     generate_negative_non_faces(
#         character_annotations_file, character_folder, output_negative_faces_folder)


# def copy_images(source_folder, destination_folder, character_name):
#     os.makedirs(destination_folder, exist_ok=True)

#     for root, _, files in os.walk(source_folder):
#         for file in files:
#             source_path = os.path.join(root, file)

#             title, ext = os.path.splitext(file)
#             new_title = f'{title}_{character_name}{ext}'

#             destination_path = os.path.join(destination_folder, new_title)
#             shutil.copy(source_path, destination_path)


# positive_images_folder = os.path.join(EXAMPLES, 'imaginiPozitive')
# negative_images_folder = os.path.join(EXAMPLES, 'imaginiNegative')
# negative_faces_images_folder = os.path.join(EXAMPLES, 'imaginiNegative')


# for character in ['barney', 'betty', 'fred', 'wilma']:
#     positive_folder = os.path.join(
#         FACES, f'{character}_imagini_pozitive')
#     copy_images(positive_folder, positive_images_folder, character)

#     negative_folder = os.path.join(
#         FACES, f'{character}_imagini_negative')
#     copy_images(negative_folder, negative_images_folder, character)

#     negative_faces_folder = os.path.join(
#         FACES, f'{character}_non_fete')
#     copy_images(negative_faces_folder, negative_faces_images_folder, character)


facial_detector = FacialDetector()

positive_features_path = os.path.join(
    'descriptori', 'descriptori_pozitivi.npy')
if os.path.exists(positive_features_path):
    positive_descriptors = np.load(positive_features_path)
    print(positive_descriptors.shape)
    print('Descriptori pozitivi incarcati')
else:
    print('Se genereaza descriptori pozitivi...')
    positive_descriptors = facial_detector.get_positive_descriptors()
    print(positive_descriptors.shape)
    np.save(positive_features_path, positive_descriptors)

negative_features_path = os.path.join(
    'descriptori', 'descriptori_negativi.npy')
if os.path.exists(negative_features_path):
    negative_descriptors = np.load(negative_features_path)
    print(negative_descriptors.shape)
    print('Descriptori negativi incarcati')
else:
    print('Se genereaza descriptori negativi...')
    negative_descriptors = facial_detector.get_negative_descriptors()
    print(negative_descriptors.shape)
    np.save(negative_features_path, negative_descriptors)


training_examples = np.concatenate(
    (np.squeeze(positive_descriptors), np.squeeze(negative_descriptors)), axis=0)
print("Numarul de caracteristici folosite in modelul de antrenare:",
      training_examples.shape[1])

train_labels = np.concatenate(
    (np.ones(positive_descriptors.shape[0]), np.zeros(negative_descriptors.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)


detections, scores, file_names = facial_detector.run()
# eval_detections('validare/task1_gt_validare1.txt',
#                 'rezultat', detections, scores, file_names)

np.save(FINAL_RESULTS_TASK1 + 'detections_all_faces.npy', detections)
np.save(FINAL_RESULTS_TASK1 + 'scores_all_faces.npy', scores)
np.save(FINAL_RESULTS_TASK1 + 'file_names_all_faces.npy', file_names)

show_detections_without_ground_truth(detections, scores, file_names)
