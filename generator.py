from imports import *
from FacialDetector import *
from utils import *
from evals import *
from params import *


def generate_positive_faces(character_annotations_file, character_folder, output_positive_folder):
    faces_by_image = extract_faces(character_annotations_file)

    for image_filename, faces in faces_by_image.items():
        image_path = os.path.join(character_folder, image_filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        for face_idx, face_box in enumerate(faces):
            xmin, ymin, xmax, ymax = face_box
            face = image[ymin:ymax, xmin:xmax]

            face = cv2.resize(face, (64, 64))

            output_positive_filename = os.path.join(
                output_positive_folder, f'{image_filename}_face{face_idx + 1}.jpg')
            cv2.imwrite(output_positive_filename, face)


def generate_negative_faces(character_annotations_file, character_folder, output_negative_folder, num_non_faces=3):
    faces_by_image = extract_faces(character_annotations_file)

    for image_filename, faces in faces_by_image.items():
        image_path = os.path.join(character_folder, image_filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        for _ in range(num_non_faces):
            xmin = np.random.randint(0, width - 64)
            ymin = np.random.randint(0, height - 64)
            xmax = xmin + 64
            ymax = ymin + 64

            intersects_positive = any(
                intersect(xmin, ymin, xmax, ymax, *face)
                for face in faces
            )

            while intersects_positive:
                xmin = np.random.randint(0, width - 64)
                ymin = np.random.randint(0, height - 64)
                xmax = xmin + 64
                ymax = ymin + 64

                intersects_positive = any(
                    intersect(xmin, ymin, xmax, ymax, *face)
                    for face in faces
                )

            non_face = image[ymin:ymax, xmin:xmax]
            output_negative_filename = os.path.join(
                output_negative_folder, f'{image_filename}_non_face{_ + 1}.jpg')
            cv2.imwrite(output_negative_filename, non_face)


def intersect(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    return not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2)


def generate_negative_non_faces(character_annotations_file, character_folder, output_negative_folder, num_non_faces=3, tolerance=20):
    faces_by_image = extract_faces(character_annotations_file)

    for image_filename, faces in faces_by_image.items():
        image_path = os.path.join(character_folder, image_filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        for _ in range(num_non_faces):
            ymin = np.random.randint(height // 2, height - 64)
            xmin = np.random.randint(0, width - 64)
            xmax = xmin + 64
            ymax = ymin + 64

            intersects_positive = any(
                intersect(xmin, ymin, xmax, ymax, *face)
                for face in faces
            )

            while intersects_positive:
                ymin = np.random.randint(height // 2, height - 64)
                xmin = np.random.randint(0, width - 64)
                xmax = xmin + 64
                ymax = ymin + 64

                intersects_positive = any(
                    intersect(xmin, ymin, xmax, ymax, *face)
                    for face in faces
                )

            patch = image[ymin:ymax, xmin:xmax]

            if has_approximate_skin_color(patch, tolerance):
                output_negative_filename = os.path.join(
                    output_negative_folder, f'{image_filename}_non_face{_ + 1}.jpg')
                cv2.imwrite(output_negative_filename, patch)

                augment_and_save(patch, image_filename,
                                 output_negative_folder, _ + 1)


def augment_and_save(patch, image_filename, output_folder, index):
    rotation_angles = [90]

    for angle in rotation_angles:
        rotated_patch = rotate_image(patch, angle)
        rotated_filename = f'{image_filename}_non_face{index}_rotated_{angle}.jpg'
        output_rotated_filename = os.path.join(output_folder, rotated_filename)
        cv2.imwrite(output_rotated_filename, rotated_patch)


def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return rotated_image


def has_approximate_skin_color(patch, tolerance):
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)

    mean_lab_color = np.mean(lab_patch, axis=(0, 1))

    skin_tone_color = np.array([150, 128, 128])

    color_difference = np.linalg.norm(mean_lab_color - skin_tone_color)
    return color_difference < tolerance


def generate_positive_faces_per_character(character_annotations_file1, character_annotations_file2, character_folder1, character_folder2, output_positive_folder1, output_positive_folder2, character1, character2, output_negative_folder1, output_negative_folder2):
    with open(character_annotations_file1, 'r') as file1, open(character_annotations_file2, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            if line1.strip() and line2.strip():
                values1 = line1.strip().split()
                values2 = line2.strip().split()

                if len(values1) >= 6 and len(values2) >= 6:
                    image_name1, xmin1, ymin1, xmax1, ymax1, character_name1 = values1[:6]
                    image_name2, xmin2, ymin2, xmax2, ymax2, character_name2 = values2[:6]

                    image_path1 = os.path.join(character_folder1, image_name1)
                    image_path2 = os.path.join(character_folder2, image_name2)

                    image1 = cv2.imread(image_path1)
                    image2 = cv2.imread(image_path2)

                    height1, width1, _ = image1.shape
                    height2, width2, _ = image2.shape

                    xmin1, ymin1, xmax1, ymax1 = map(
                        int, [xmin1, ymin1, xmax1, ymax1])
                    xmin2, ymin2, xmax2, ymax2 = map(
                        int, [xmin2, ymin2, xmax2, ymax2])

                    face1 = image1[ymin1:ymax1, xmin1:xmax1]
                    face2 = image2[ymin2:ymax2, xmin2:xmax2]

                    face1 = cv2.resize(face1, (64, 64))
                    face2 = cv2.resize(face2, (64, 64))

                    if character_name1 == character1 and character_name2 == character2:
                        output_positive_filename1 = os.path.join(
                            output_positive_folder1, f'{image_name1}_face{len(values1) + 1}.jpg')
                        output_positive_filename2 = os.path.join(
                            output_positive_folder2, f'{image_name2}_face{len(values2) + 1}.jpg')

                        output_negative_filename1 = os.path.join(
                            output_negative_folder1, f'{image_name2}_face{len(values2) + 1}.jpg')
                        output_negative_filename2 = os.path.join(
                            output_negative_folder2, f'{image_name1}_face{len(values1) + 1}.jpg')

                        cv2.imwrite(output_positive_filename1, face1)
                        cv2.imwrite(output_positive_filename2, face2)
                        cv2.imwrite(output_negative_filename1, face2)
                        cv2.imwrite(output_negative_filename2, face1)
